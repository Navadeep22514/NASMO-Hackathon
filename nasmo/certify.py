"""Emit machine-checkable optimality certificates.

Two files are written:

* ``front.json``      - the complete Pareto front (all 5 objectives per
                        point) plus each point's architecture spec.
* ``coverage.json``   - the evidence that the front is COMPLETE: certified
                        cutoff D*, lemma parameters, the raw integer layer
                        table, per-round frontier sizes + hashes, and
                        hashes of the profile/space inputs.

``nasmo.verify`` independently re-derives everything and refuses to
endorse the certificate if any part was tampered with.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from . import OBJECTIVE_ORDER
from .models import Problem, State, reported_objectives
from .search import Front

FRONT_FORMAT = "nasmo-front-v1"
COVERAGE_FORMAT = "nasmo-coverage-v1"


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def points_digest(points: List[dict]) -> str:
    """Canonical digest of the point list; recomputed by the verifier."""
    h = hashlib.sha256()
    for p in points:
        obj = ",".join(f"{k}={p['objectives'][k]}" for k in OBJECTIVE_ORDER)
        arch = ">".join(p["arch"])
        h.update(f"{p['id']}|{p['depth']}|{arch}|{','.join(map(str, p['state']))}|{obj}\n".encode())
    return h.hexdigest()


def build_points(prob: Problem, front: Front) -> List[dict]:
    """Deterministically ordered front points: best accuracy first, then
    cheapest, then lexicographic architecture."""
    rows = []
    for state, arch in front.items():
        obj = reported_objectives(state, prob.profile, prob.am)
        rows.append((obj, arch, state))
    rows.sort(key=lambda r: (-r[0]["accuracy_bp"], r[0]["latency_us"],
                             r[0]["energy_uj"], r[0]["peak_mem_bytes"],
                             r[0]["fairness_bp"], r[1]))
    points = []
    for i, (obj, arch, state) in enumerate(rows, start=1):
        points.append({"id": f"P{i:04d}", "depth": len(arch),
                       "arch": list(arch), "state": list(state),
                       "objectives": obj})
    return points


def certify(prob: Problem, front: Front, rounds: List[dict], meta: dict,
            cert_dir, profile_path, space_path) -> Dict[str, Path]:
    cert_dir = Path(cert_dir)
    cert_dir.mkdir(parents=True, exist_ok=True)
    points = build_points(prob, front)

    front_doc = {
        "format": FRONT_FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "space_name": prob.space["space_name"],
        "device": prob.profile["device"],
        "objective_order": OBJECTIVE_ORDER,
        "points": points,
    }
    coverage_doc = {
        "format": COVERAGE_FORMAT,
        "generated_utc": front_doc["generated_utc"],
        "stop_reason": meta["stop_reason"],
        "stop_depth": meta["stop_depth"],
        "d_star": meta["d_star"],
        "front_size": meta["front_size"],
        "lemma": prob.lemma,
        "layer_type_count": len(prob.types),
        "layer_types": [t._asdict() for t in prob.types],
        "rounds": rounds,
        "points_digest": points_digest(points),
        "profile_sha256": file_sha256(profile_path),
        "space_sha256": file_sha256(space_path),
        "search": {"algorithm": "pareto-dp-f-extension",
                   "python": __import__("sys").version.split()[0],
                   "package_version": __import__("nasmo").__version__},
    }
    front_path = cert_dir / "front.json"
    coverage_path = cert_dir / "coverage.json"
    front_path.write_text(json.dumps(front_doc, indent=1), encoding="utf-8")
    coverage_path.write_text(json.dumps(coverage_doc, indent=1), encoding="utf-8")
    return {"front": front_path, "coverage": coverage_path}
