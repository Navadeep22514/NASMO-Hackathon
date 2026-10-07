"""Independent verifier for NASMO optimality certificates.

The verifier trusts NOTHING inside the certificate except as a claim to be
re-derived.  For a directory containing ``front.json`` + ``coverage.json`` it
re-loads the raw search-space and hardware-profile files, re-computes their
SHA-256 digests, re-derives the cutoff lemma, re-builds every layer table
row, re-computes every architecture's state vector and objectives, checks
that no point on the claimed Pareto front dominates another, checks the
coverage/stop-rule evidence, and finally re-runs the search from scratch
and compares the resulting frontier and per-round hashes bit-for-bit.

Any tampering with a single number makes at least one check fail.

CLI::

    python -m nasmo.verify [cert_dir] [--profile P] [--space S] [--no-rerun]

Exit status 0 iff every check passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

from . import OBJECTIVE_ORDER, SENSE
from .certify import (FRONT_FORMAT, COVERAGE_FORMAT, build_points,
                      points_digest)
from .models import (Problem, arch_state, reported_objectives,
                     objectives_dominates, load_profile, load_space,
                     DEFAULT_PROFILE, DEFAULT_SPACE)
from .search import _vec_hash, run_search
from .space import validate_arch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CERT_DIR = ROOT / "certs"


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class Verifier:
    def __init__(self) -> None:
        self.checks: List[dict] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"name": name, "ok": bool(ok),
                            "detail": detail if not ok else ""})
        return ok

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks)

    def result(self) -> dict:
        return {"ok": self.ok,
                "passed": sum(1 for c in self.checks if c["ok"]),
                "total": len(self.checks),
                "checks": self.checks}


def verify(cert_dir=DEFAULT_CERT_DIR, profile_path=DEFAULT_PROFILE,
           space_path=DEFAULT_SPACE, rerun_search: bool = True) -> dict:
    """Verify the certificate in ``cert_dir``. Returns a result dict."""
    v = Verifier()
    cert_dir = Path(cert_dir)

    # ---------------------------------------------------------- load files
    front_path = cert_dir / "front.json"
    cov_path = cert_dir / "coverage.json"
    if not v.check("certificate files exist",
                   front_path.is_file() and cov_path.is_file(),
                   f"missing {front_path if not front_path.is_file() else cov_path}"):
        return v.result()
    try:
        front_doc = json.loads(front_path.read_text(encoding="utf-8"))
        cov_doc = json.loads(cov_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        v.check("certificate parses as JSON", False, str(e))
        return v.result()

    v.check("front format tag", front_doc.get("format") == FRONT_FORMAT,
            f"expected {FRONT_FORMAT!r} got {front_doc.get('format')!r}")
    v.check("coverage format tag", cov_doc.get("format") == COVERAGE_FORMAT,
            f"expected {COVERAGE_FORMAT!r} got {cov_doc.get('format')!r}")
    v.check("objective order", front_doc.get("objective_order") == OBJECTIVE_ORDER,
            f"got {front_doc.get('objective_order')}")
    points = front_doc.get("points") or []
    v.check("front is non-empty", len(points) > 0, "0 points")

    # ------------------------------------------- raw inputs still intact?
    actual_profile_sha = _sha256(profile_path)
    actual_space_sha = _sha256(space_path)
    v.check("profile file sha256 matches certificate",
            actual_profile_sha == cov_doc.get("profile_sha256"),
            f"actual {actual_profile_sha} vs claimed {cov_doc.get('profile_sha256')}")
    v.check("space file sha256 matches certificate",
            actual_space_sha == cov_doc.get("space_sha256"),
            f"actual {actual_space_sha} vs claimed {cov_doc.get('space_sha256')}")

    # ---------------------------------- re-derive problem from raw inputs
    try:
        prob = Problem(load_profile(profile_path), load_space(space_path))
    except Exception as e:  # malformed inputs
        v.check("raw inputs rebuild a problem", False, repr(e))
        return v.result()

    v.check("cutoff lemma re-derived identically",
            prob.lemma == cov_doc.get("lemma"),
            f"recomputed {prob.lemma} vs claimed {cov_doc.get('lemma')}")
    claimed_types = cov_doc.get("layer_types") or []
    actual_types = [t._asdict() for t in prob.types]
    v.check("layer-type table re-derived identically",
            actual_types == claimed_types,
            f"recomputed {len(actual_types)} rows vs claimed {len(claimed_types)}")
    v.check("layer type count", cov_doc.get("layer_type_count") == len(prob.types),
            f"{cov_doc.get('layer_type_count')} vs {len(prob.types)}")

    # --------------------------------------------- points digest matches?
    try:
        digest = points_digest(points)
    except Exception as e:
        v.check("points digest computable", False, repr(e))
        digest = ""
    v.check("points digest matches coverage",
            digest == cov_doc.get("points_digest"),
            f"recomputed {digest} vs claimed {cov_doc.get('points_digest')}")

    # --------------------------------- every point internally consistent?
    bad_arch = bad_depth = bad_state = bad_obj = 0
    for p in points:
        arch = p.get("arch") or []
        try:
            validate_arch(arch, prob.types_by_id)
        except ValueError:
            bad_arch += 1
            continue
        if p.get("depth") != len(arch):
            bad_depth += 1
            continue
        state = tuple(p.get("state") or [])
        try:
            recomputed = arch_state(arch, prob.types_by_id, prob.c_sat)
        except Exception:
            bad_state += 1
            continue
        if state != recomputed:
            bad_state += 1
            continue
        if reported_objectives(state, prob.profile, prob.am) != p.get("objectives"):
            bad_obj += 1
    v.check("every architecture is in the search space", bad_arch == 0,
            f"{bad_arch} invalid architectures")
    v.check("depth equals architecture length", bad_depth == 0,
            f"{bad_depth} mismatched depths")
    v.check("state vectors recompute from architectures", bad_state == 0,
            f"{bad_state} states do not match their architecture")
    v.check("objectives recompute from state vectors", bad_obj == 0,
            f"{bad_obj} objective vectors do not match")

    # --------------------------------- front points mutually non-dominated
    dominated_pairs = 0
    objs = [p.get("objectives") for p in points]
    for i in range(len(objs)):
        for j in range(len(objs)):
            if i != j and objs[i] and objs[j] and objectives_dominates(objs[i], objs[j]):
                dominated_pairs += 1
                break
    v.check("no front point dominates another", dominated_pairs == 0,
            f"{dominated_pairs} dominated points on the claimed front")

    # ---------------------------------------------------- coverage record
    v.check("declared front size matches point count",
            cov_doc.get("front_size") == len(points),
            f"{cov_doc.get('front_size')} vs {len(points)}")
    v.check("declared D* matches re-derived cutoff lemma",
            cov_doc.get("d_star") == prob.lemma["d_star"],
            f"{cov_doc.get('d_star')} vs {prob.lemma['d_star']}")
    rounds = cov_doc.get("rounds") or []
    v.check("coverage has rounds", len(rounds) > 0, "empty round log")
    stop = cov_doc.get("stop_reason") or "?"
    stop_depth = cov_doc.get("stop_depth")
    if rounds and stop:
        if stop == "fixed_point":
            last_ok = rounds[-1].get("fixed_point") is True
            v.check("stop rule: fixed point recorded on last round", last_ok,
                    f"last round {rounds[-1]}")
            v.check("stop rule: stop depth equals last round depth",
                    stop_depth == rounds[-1].get("depth"),
                    f"{stop_depth} vs {rounds[-1].get('depth')}")
            v.check("round depths are consecutive from 1",
                    [r.get("depth") for r in rounds] == list(range(1, len(rounds) + 1)),
                    f"got {[r.get('depth') for r in rounds[:8]]}...")
        elif stop == "cutoff_bound":
            v.check("stop rule: cutoff depth equals D*",
                    stop_depth == prob.lemma["d_star"],
                    f"{stop_depth} vs {prob.lemma['d_star']}")
        else:
            v.check("stop rule recognised", False, f"unknown {stop!r}")
        v.check("stop depth within certified cutoff D*",
                isinstance(stop_depth, int) and 1 <= stop_depth <= prob.lemma["d_star"],
                f"stop_depth={stop_depth} D*={prob.lemma['d_star']}")

    # -------------------------------- independent re-run of the whole search
    if rerun_search:
        re_front, re_rounds, re_meta = run_search(prob)
        re_points_digest = points_digest(
            build_points(prob, re_front))
        v.check("re-run: same stop reason",
                re_meta["stop_reason"] == stop,
                f"{re_meta['stop_reason']} vs {stop}")
        v.check("re-run: same frontier size",
                re_meta["front_size"] == len(points),
                f"{re_meta['front_size']} vs {len(points)}")
        v.check("re-run: same points digest",
                re_points_digest == cov_doc.get("points_digest"),
                f"{re_points_digest} vs {cov_doc.get('points_digest')}")
        re_hashes = [r["hash"] for r in re_rounds]
        claimed_hashes = [r["hash"] for r in rounds]
        v.check("re-run: identical per-round frontier hashes",
                re_hashes == claimed_hashes,
                f"recomputed {len(re_hashes)} round hashes vs claimed {len(claimed_hashes)}")
        re_states = set(re_front.keys())
        claimed_states = {tuple(p["state"]) for p in points if "state" in p}
        v.check("re-run: identical frontier state sets",
                re_states == claimed_states,
                f"{len(re_states)} vs {len(claimed_states)} states")

    return v.result()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify a NASMO optimality certificate")
    ap.add_argument("cert_dir", nargs="?", default=str(DEFAULT_CERT_DIR))
    ap.add_argument("--profile", default=str(DEFAULT_PROFILE))
    ap.add_argument("--space", default=str(DEFAULT_SPACE))
    ap.add_argument("--no-rerun", action="store_true",
                    help="skip the full independent re-run (faster, weaker)")
    args = ap.parse_args(argv)

    print(f"NASMO certificate verifier — checking {args.cert_dir}")
    res = verify(args.cert_dir, args.profile, args.space,
                 rerun_search=not args.no_rerun)
    for c in res["checks"]:
        mark = "PASS" if c["ok"] else "FAIL"
        line = f"  [{mark}] {c['name']}"
        if c["detail"]:
            line += f" :: {c['detail']}"
        print(line)
    print(f"\n{res['passed']}/{res['total']} checks passed")
    if res["ok"]:
        print("CERTIFICATE VALID")
        return 0
    print("CERTIFICATE INVALID")
    return 1


if __name__ == "__main__":
    sys.exit(main())
