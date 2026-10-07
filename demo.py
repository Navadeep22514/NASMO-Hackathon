#!/usr/bin/env python3
"""NASMO end-to-end demo — one command, full pipeline.

    python demo.py [--cert-dir DIR] [--report PATH] [--no-rerun]

Steps
  1. load the hardware profile + search space,
  2. run the certified multi-objective search (5 objectives),
  3. emit the machine-checkable certificate (front.json + coverage.json),
  4. independently verify every claim in the certificate (re-derives the
     cutoff lemma, layer table, every state/objective, mutual non-
     domination, coverage log, and re-runs the entire search),
  5. generate the hardware-aware performance report bound to the
     certificate digests.

Exit status 0 iff the certificate verified.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from nasmo.certify import certify
from nasmo.models import Problem, DEFAULT_PROFILE, DEFAULT_SPACE
from nasmo.report import generate_report
from nasmo.search import run_search
from nasmo.verify import verify

ROOT = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NASMO end-to-end demo")
    ap.add_argument("--cert-dir", default=str(ROOT / "certs"))
    ap.add_argument("--report", default=str(ROOT / "report.html"))
    ap.add_argument("--no-rerun", action="store_true",
                    help="verifier skips the independent search re-run (faster)")
    args = ap.parse_args(argv)

    print("=" * 72)
    print(" NASMO — Neural Architecture Search with Machine-checkable Optimality")
    print("=" * 72)

    t0 = time.time()
    print("\n[1/5] loading hardware profile + search space ...")
    prob = Problem.load()
    print(f"      device       : {prob.profile['device']}")
    print(f"      layer types  : {len(prob.types)}")
    print(f"      objectives   : {', '.join(prob.space['objectives'][i]['name'] for i in range(5))}")
    print(f"      cutoff D*    : {prob.lemma['d_star']} (certified bound for the unbounded space)")

    print("\n[2/5] running certified Pareto search ...")
    t1 = time.time()
    front, rounds, meta = run_search(prob)
    t2 = time.time()
    print(f"      frontier     : {meta['front_size']} globally optimal architectures")
    print(f"      stop         : {meta['stop_reason']} at depth {meta['stop_depth']} / D*={meta['d_star']}")
    print(f"      search time  : {t2 - t1:.2f}s over {len(rounds)} rounds")

    print("\n[3/5] emitting machine-checkable certificate ...")
    paths = certify(prob, front, rounds, meta, args.cert_dir,
                    DEFAULT_PROFILE, DEFAULT_SPACE)
    for k, p in paths.items():
        print(f"      {k:<10}-> {p}")

    print("\n[4/5] independent verification (verifier re-derives everything) ...")
    res = verify(args.cert_dir, rerun_search=not args.no_rerun)
    for c in res["checks"]:
        mark = "PASS" if c["ok"] else "FAIL"
        line = f"      [{mark}] {c['name']}"
        if c["detail"]:
            line += f" :: {c['detail']}"
        print(line)
    verdict = "CERTIFICATE VALID" if res["ok"] else "CERTIFICATE INVALID"
    print(f"      -> {verdict} ({res['passed']}/{res['total']} checks)")

    print("\n[5/5] hardware-aware performance report ...")
    import json
    coverage = json.loads(Path(args.cert_dir, "coverage.json").read_text(encoding="utf-8"))
    points = json.loads(Path(args.cert_dir, "front.json").read_text(encoding="utf-8"))["points"]
    rep = generate_report(prob, points, meta, coverage, res, args.report)
    print(f"      report       -> {rep}")

    print(f"\nTotal: {time.time() - t0:.2f}s")
    print("=" * 72)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
