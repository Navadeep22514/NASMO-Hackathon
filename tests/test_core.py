"""NASMO core + certificate-tamper tests (no framework needed).

    python tests/test_core.py

Exit 0 iff everything passes.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nasmo.models import (Problem, accuracy_bp, cutoff_lemma,
                          objectives_dominates, reported_objectives,
                          DEFAULT_PROFILE)
from nasmo.search import run_search
from nasmo.space import make_layer_id, parse_layer_id
from nasmo.verify import verify

PASSED = 0


def ok(cond: bool, name: str) -> None:
    global PASSED
    if not cond:
        print(f"FAIL {name}")
        sys.exit(1)
    PASSED += 1
    print(f"ok   {name}")


def main() -> int:
    prob = Problem.load()

    # ------------------------------------------------------------ unit level
    ok(parse_layer_id(make_layer_id("sepconv3", 1.5, True)) == ("sepconv3", 1.5, True),
       "layer id roundtrip")
    ok(prob.lemma["d_star"] == 29, "cutoff lemma D* = 29")
    ok(prob.lemma["a_min"] > 0, "every layer adds capacity (a_min > 0)")

    # accuracy strictly increasing below saturation, flat above
    am = prob.am
    caps = [0, 100, am["capacity_saturation"] - 1, am["capacity_saturation"],
            am["capacity_saturation"] + 500]
    accs = [accuracy_bp(c, am) for c in caps]
    ok(all(a < b for a, b in zip(accs[:3], accs[1:4])),
       "accuracy strictly increases below c_sat")
    ok(accs[3] == accs[4] == am["acc_max_bp"], "accuracy saturates at c_sat")

    # dominance semantics
    a = {"accuracy_bp": 90, "latency_us": 10, "energy_uj": 10,
         "peak_mem_bytes": 10, "fairness_bp": 10}
    b = dict(a, latency_us=20)          # a strictly better on latency
    c = dict(a, accuracy_bp=80)         # a better accuracy, worse nothing... c worse on acc
    ok(objectives_dominates(a, b), "strictly better cost dominates")
    ok(not objectives_dominates(b, a), "dominance is antisymmetric")
    ok(not objectives_dominates(a, a), "no self-dominance")
    ok(objectives_dominates(a, c), "better accuracy dominates")
    ok(not objectives_dominates(c, a), "worse accuracy does not dominate")

    # ------------------------------------------- full search + certificate
    cert_dir = ROOT / "certs"
    ok((cert_dir / "front.json").is_file(), "certificate exists (run demo.py first)")
    front, rounds, meta = run_search(prob)
    ok(meta["front_size"] == 2274, "frontier has 2274 points")
    ok(meta["stop_reason"] == "fixed_point", "search stops at a fixed point")
    ok(meta["stop_depth"] <= prob.lemma["d_star"], "stop depth within certified D*")

    # --------------------------------------------------- verify (re-runs all)
    res = verify(cert_dir, rerun_search=True)
    ok(res["ok"], f"certificate verifies ({res['passed']}/{res['total']})")

    # ------------------------------------------- tamper test 1: edit a number
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        shutil.copy(cert_dir / "front.json", tdir)
        shutil.copy(cert_dir / "coverage.json", tdir)
        doc = json.loads((tdir / "front.json").read_text(encoding="utf-8"))
        doc["points"][0]["objectives"]["accuracy_bp"] += 1  # forge a better accuracy
        (tdir / "front.json").write_text(json.dumps(doc), encoding="utf-8")
        res2 = verify(tdir, rerun_search=False)             # fast: cheap checks suffice
        ok(not res2["ok"], "forged accuracy is rejected")
        failed = {c["name"] for c in res2["checks"] if not c["ok"]}
        ok("points digest matches coverage" in failed, "  -> digest check fires")
        ok("objectives recompute from state vectors" in failed,
           "  -> objective recompute check fires")

    # --------------------------------------- tamper test 2: tampered raw input
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        shutil.copy(cert_dir / "front.json", tdir)
        shutil.copy(cert_dir / "coverage.json", tdir)
        prof = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))
        prof["ops"]["sepconv3"]["latency_us"] += 1          # change the "hardware"
        ppath = tdir / "profile.json"
        ppath.write_text(json.dumps(prof), encoding="utf-8")
        res3 = verify(tdir, profile_path=ppath, rerun_search=False)
        ok(not res3["ok"], "tampered hardware profile is rejected")
        failed3 = {c["name"] for c in res3["checks"] if not c["ok"]}
        ok("profile file sha256 matches certificate" in failed3,
           "  -> profile hash check fires")

    # --------------------------------------- tamper test 3: swapped arch spec
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        shutil.copy(cert_dir / "front.json", tdir)
        shutil.copy(cert_dir / "coverage.json", tdir)
        doc = json.loads((tdir / "front.json").read_text(encoding="utf-8"))
        p = doc["points"][5]
        # swap the first layer for a different one, keeping the depth so the
        # state/objective recompute checks are the ones that must fire
        for cand in ("dilconv3@1.5R", "sepconv5@1.5N", "skip@0.5R"):
            if cand != p["arch"][0]:
                p["arch"] = [cand] + p["arch"][1:]
                break
        (tdir / "front.json").write_text(json.dumps(doc), encoding="utf-8")
        res4 = verify(tdir, rerun_search=False)
        ok(not res4["ok"], "swapped architecture is rejected")
        failed4 = {c["name"] for c in res4["checks"] if not c["ok"]}
        ok("state vectors recompute from architectures" in failed4,
           "  -> state recompute check fires")
        ok("points digest matches coverage" in failed4,
           "  -> digest check fires (arch witness is bound into the digest)")

    print(f"\n{PASSED} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
