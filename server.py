"""NASMO server — FastAPI backend with SQLite run history.

    python -m uvicorn server:app --port 8000     (or: python server.py)

API
    GET  /api/health            liveness
    GET  /api/profile           hardware profile + objectives + lemma + layer table
    POST /api/search            run the certified search, persist run to SQLite
    GET  /api/runs              run history from SQLite
    GET  /api/front/{id}        complete Pareto front (all points, 5 objectives)
    GET  /api/cert/{id}         coverage certificate (lemma, rounds, digests)
    POST /api/verify/{id}       independent verification (re-runs the search)
    GET  /api/report/{id}       hardware-aware performance report (HTML)
    GET  /                     the NASMO frontend (web/index.html)

Every endpoint returns JSON errors as {"detail": ...} with a proper status
code; the search/verify endpoints are def (threadpool) functions so the
CPU-bound work never blocks the event loop.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from nasmo.certify import certify
from nasmo.models import Problem, DEFAULT_PROFILE, DEFAULT_SPACE
from nasmo.report import generate_report
from nasmo.search import run_search
from nasmo.verify import verify

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "nasmo.db"
RUNS_DIR = ROOT / "certs" / "runs"
WEB_DIR = ROOT / "web"

app = FastAPI(title="NASMO", version="1.0.0",
              description="Explainable Multi-Objective NAS with machine-checkable "
                          "global optimality certificates")

_prob: Optional[Problem] = None


def get_problem() -> Problem:
    global _prob
    if _prob is None:
        _prob = Problem.load()
    return _prob


# ------------------------------------------------------------------ database
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_utc TEXT NOT NULL,
                label TEXT NOT NULL,
                cert_dir TEXT NOT NULL,
                stop_reason TEXT,
                stop_depth INTEGER,
                d_star INTEGER,
                front_size INTEGER,
                points_digest TEXT,
                search_seconds REAL,
                verify_json TEXT
            )""")


def run_row(run_id: int) -> sqlite3.Row:
    with db() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(404, f"run {run_id} not found")
    return row


def cert_dir_of(run_id: int) -> Path:
    d = Path(run_row(run_id)["cert_dir"])
    if not (d / "front.json").is_file():
        raise HTTPException(404, f"certificate files for run {run_id} missing")
    return d


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(500, f"cannot read {path.name}: {e}")


def seed_from_demo() -> None:
    """Register the certificate produced by `python demo.py` as run 1, so the
    dashboard has data before the first live search."""
    demo_cert = ROOT / "certs"
    if not (demo_cert / "front.json").is_file():
        return
    with db() as conn:
        n = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]
    if n:
        return
    coverage = load_json(demo_cert / "coverage.json")
    digest = coverage.get("points_digest", "")
    with db() as conn:
        conn.execute(
            "INSERT INTO runs (created_utc, label, cert_dir, stop_reason, stop_depth,"
            " d_star, front_size, points_digest, search_seconds) VALUES (?,?,?,?,?,?,?,?,?)",
            (coverage.get("generated_utc", ""), "demo (python demo.py)", str(demo_cert),
             coverage.get("stop_reason"), coverage.get("stop_depth"),
             coverage.get("d_star"), coverage.get("front_size"), digest, None))


init_db()
seed_from_demo()


# ------------------------------------------------------------------- schemas
class SearchBody(BaseModel):
    label: str = ""


class VerifyBody(BaseModel):
    rerun: bool = True


# ----------------------------------------------------------------- endpoints
@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "nasmo", "version": "1.0.0"}


@app.get("/api/profile")
def profile() -> dict:
    prob = get_problem()
    types = [t._asdict() for t in prob.types]
    return {
        "device": prob.profile,
        "space_name": prob.space["space_name"],
        "objectives": prob.space["objectives"],
        "lemma": prob.lemma,
        "layer_types": types,
        "layer_type_count": len(types),
    }


@app.post("/api/search")
def search(body: SearchBody = SearchBody()) -> dict:
    """Run the certified 5-objective search and persist the run."""
    try:
        prob = get_problem()
        t0 = __import__("time").time()
        front, rounds, meta = run_search(prob)
        secs = __import__("time").time() - t0
        run_id = None
        with db() as conn:
            cur = conn.execute(
                "INSERT INTO runs (created_utc, label, cert_dir, stop_reason,"
                " stop_depth, d_star, front_size, search_seconds) VALUES (?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 body.label or f"live search #{__import__('time').time_ns() % 10000}",
                 "", meta["stop_reason"], meta["stop_depth"], meta["d_star"],
                 meta["front_size"], round(secs, 2)))
            run_id = cur.lastrowid
        cert_dir = RUNS_DIR / f"run_{run_id}"
        paths = certify(prob, front, rounds, meta, cert_dir,
                        DEFAULT_PROFILE, DEFAULT_SPACE)
        with db() as conn:
            conn.execute("UPDATE runs SET cert_dir = ? WHERE id = ?",
                         (str(cert_dir), run_id))
        return {"id": run_id, "label": body.label or "live search",
                "stop_reason": meta["stop_reason"], "stop_depth": meta["stop_depth"],
                "d_star": meta["d_star"], "front_size": meta["front_size"],
                "rounds": len(rounds), "seconds": round(secs, 2),
                "files": {k: str(v) for k, v in paths.items()}}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"search failed: {e}")


@app.get("/api/runs")
def runs() -> dict:
    with db() as conn:
        rows = conn.execute(
            "SELECT id, created_utc, label, stop_reason, stop_depth, d_star,"
            " front_size, points_digest, search_seconds, verify_json FROM runs"
            " ORDER BY id DESC").fetchall()
    out = []
    for r in rows:
        item = dict(r)
        vj = item.pop("verify_json", None)
        item["verified"] = (json.loads(vj)["ok"] if vj else None)
        out.append(item)
    return {"runs": out}


@app.get("/api/front/{run_id}")
def front(run_id: int) -> dict:
    doc = load_json(cert_dir_of(run_id) / "front.json")
    return {"run_id": run_id, **doc}


@app.get("/api/cert/{run_id}")
def cert(run_id: int) -> dict:
    doc = load_json(cert_dir_of(run_id) / "coverage.json")
    stored = run_row(run_id)["verify_json"]
    return {"run_id": run_id, "coverage": doc,
            "verify": json.loads(stored) if stored else None}


@app.post("/api/verify/{run_id}")
def verify_run(run_id: int, body: VerifyBody = VerifyBody()) -> dict:
    try:
        res = verify(cert_dir_of(run_id), rerun_search=body.rerun)
        with db() as conn:
            conn.execute("UPDATE runs SET verify_json = ? WHERE id = ?",
                         (json.dumps(res), run_id))
        return {"run_id": run_id, **res}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"verification failed: {e}")


@app.get("/api/report/{run_id}")
def report(run_id: int) -> HTMLResponse:
    row = run_row(run_id)
    cert_dir = cert_dir_of(run_id)
    prob = get_problem()
    front_doc = load_json(cert_dir / "front.json")
    coverage = load_json(cert_dir / "coverage.json")
    vres = json.loads(row["verify_json"]) if row["verify_json"] else None
    meta = {k: coverage.get(k) for k in ("stop_reason", "stop_depth", "d_star",
                                         "front_size")}
    out = cert_dir / "report.html"
    try:
        generate_report(prob, front_doc["points"], meta, coverage, vres, out)
        return HTMLResponse(out.read_text(encoding="utf-8"))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"report generation failed: {e}")


@app.get("/")
def index() -> FileResponse:
    idx = WEB_DIR / "index.html"
    if not idx.is_file():
        raise HTTPException(404, "frontend not found (web/index.html)")
    return FileResponse(idx, media_type="text/html")


def main() -> None:
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run("server:app", host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
