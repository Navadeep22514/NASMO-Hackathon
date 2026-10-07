# NASMO — Neural Architecture Search with Machine-checkable Optimality

**Explainable Multi-Objective Neural Architecture Search that finds the globally
optimal architecture for any objective combination — with a machine-checkable
certificate of optimality over a search space of unbounded size.**

Hackathon prototype · Domain: AI/ML

---

## 1. Problem understanding

Neural architecture search (NAS) is combinatorial: an architecture is a sequence
of layers chosen from a set of operations, widths, and connection patterns, so
even a 20-layer space has trillions of candidates — and NASMO requires **unbounded** depth. On top of that, practitioners do not care about
accuracy alone: latency, energy, peak memory, and fairness conflict with each
other, so there is no single "best" model — there is a **Pareto front** of
mutually non-dominated trade-offs. Finally, search heuristics (evolution,
reinforcement learning, Bayesian optimization) only produce *candidates*: they
cannot prove that nothing better exists, so every architecture decision they
make is unverified.

**NASMO's answer:** **Exact** Pareto dynamic program over the unbounded
layer-sequence space that returns the **complete** front for any number of
objectives, plus a **machine-checkable certificate** (`front.json` +
`coverage.json`) and an **independent verifier** that re-derives every claim
from the raw inputs — re-running the full search and comparing per-round
frontier hashes bit-for-bit.

## 2. Target users & real-world needs

| User | Need |
|---|---|
| Edge/ML engineers shipping to phones & Pi-class devices | "Give me the most accurate model under 3 ms and 1.5 MB" — hard deployment constraints |
| Fairness-sensitive deployments (health, lending, vision) | Accuracy-vs-disparity trade-off must be explicit, not hidden in a weighted score |
| Green-AI / energy-constrained teams | Energy as a first-class objective, measured per device profile |
| Researchers & auditors | Results they can *trust*: reproducible, verifiable, with a proof of optimality |

Requirements distilled from the statement: ≥ 5 simultaneous objectives ✓,
arbitrary objective combinations ✓, complete Pareto front ✓, machine-checkable
optimality certificate ✓, unbounded-size search space ✓, hardware-aware
performance report matching the certificates ✓.

## 3. System architecture / workflow

```
┌──────────────┐   objectives/sliders   ┌──────────────────────┐
│  Frontend     │ ─────────────────────▶ │  FastAPI backend      │
│  web/index.   │ ◀───────────────────── │  server.py            │
│  html (SVG +  │   front, explain,      │   ├─ search engine    │
│  canvas, no   │   certificate state    │   │  (nasmo/search)   │
│  build step)  │                        │   ├─ certify          │
└──────────────┘                        │   ├─ verify           │
                                         │   └─ report           │
                                         │  SQLite (nasmo.db)    │
                                         │  run history + verdicts│
                                         └──────────┬───────────┘
                                                    │
                    ┌───────────────────────────────┴──────────────┐
                    │  nasmo core (pure Python, exact integers)     │
                    │  space.py  → layer-type enumeration           │
                    │  models.py → state vector, accuracy model,    │
                    │              cutoff lemma D*                  │
                    │  search.py → Pareto DP + skyline (exact)      │
                    │  certify.py→ front.json + coverage.json       │
                    │  verify.py → independent re-derivation (28)   │
                    │  report.py → hardware-aware HTML report       │
                    └──────────────────────────────────────────────┘
        inputs: profiles/desktop_cpu.json (device)   config/space.json (space)
                ── both SHA-256-pinned inside every certificate ──
```

**End-to-end flow:** user sets objective constraints → backend runs the
certified search (~2 s) → certificate written & persisted (SQLite) →
independent verifier re-derives everything and stamps the run → frontend
renders the front (scatter + parallel coordinates), explains the selected
architecture, and links the verification verdict → hardware-aware report
available per run.

## 4. Why the front is globally complete (proof sketch)

Full proofs: [`docs/THEORY.md`](docs/THEORY.md).

1. **Monotone reporting** — all five objectives are monotone functions of an
   exact-integer state vector, so state-space dominance is sound for the
   reported objectives (pruning never hides a real optimum).
2. **Cutoff lemma** — every layer contributes ≥ `a_min > 0` capacity and
   accuracy saturates at `c_sat`, so any architecture deeper than
   `D* = ceil((c_sat + a_max)/a_min)` contains a prefix of depth ≤ D* that
   dominates-or-equals it. **This is what makes completeness over an
   unbounded space provable** (current config: D* = 29).
3. **Pareto DP exactness** — the search extends only frontier states; any
   dropped candidate is dominated by a stored one, and domination is
   preserved under extension.
4. **Fixed-point stopping** — once `G_d == G_{d-1}` the frontier can never
   change again (proved by induction in THEORY.md §5); the run stops at
   depth 21 with D* = 29 as fallback bound. Every round's frontier hash is
   recorded in `coverage.json`.

## 5. Technology choices

| Layer | Choice | Why |
|---|---|---|
| Search core | Pure Python, exact integers | bit-exact dominance tests; no float nondeterminism in certificates; zero heavy deps for the judge laptop |
| Proof layer | Self-contained lemma + hashes (no solver) | checkable in milliseconds by our verifier, honest about scope (space + models are hashed) |
| Backend | FastAPI + uvicorn | matches the deck's stated plan; auto threadpool keeps the 2 s search off the event loop |
| Database | SQLite (stdlib `sqlite3`) | zero-setup persistence of runs, verdicts, digests — file you can hold up to the judges |
| Frontend | Single-file HTML + SVG/canvas, vanilla JS | no build step, no CDN/internet risk at demo time; instant load |
| Report | Self-contained HTML | printable, embeds digests + full 2,274-point front |

## 6. Deliverables → evidence

| Required deliverable | Where |
|---|---|
| Multi-objective NAS pipeline with global optimality certificates | `nasmo/` pipeline; `python demo.py` → `certs/front.json` + `certs/coverage.json` |
| Complete Pareto front for ≥ 5 simultaneous objectives | 2,274 points × {accuracy, latency, energy, peak memory, fairness} |
| Formal proof that no better architecture exists | `docs/THEORY.md` + `python -m nasmo.verify certs` → **28/28 CERTIFICATE VALID** (independent re-run of the whole search) |
| Hardware-aware performance report matching the certificates | `report.html` / `GET /api/report/{id}` — device profile, lemma, digests, full front, verification stamp |

## 7. Run it

```bash
pip install fastapi uvicorn            # one-time
python demo.py                         # search + certificate + verify + report (~5 s)
python -m nasmo.verify certs           # independent check → CERTIFICATE VALID
python tests/test_core.py              # 23 checks incl. 3 tamper-rejection tests
python server.py 8765                  # dashboard at http://127.0.0.1:8765/
```

**Judge demo script (≈60 s):** click *Run certified search* → watch the
certified search + live independent verification (~5 s, green 28/28 verdict)
→ drag the fairness/latency sliders and watch the front re-rank with the
explanation panel → click *Verify certificate* again → open *Performance
report*.

## 8. Testing (Phase 2 evidence)

`tests/test_core.py` — 23 assertions, exit code 0/1:
- unit: dominance semantics, accuracy saturation monotonicity, cutoff lemma (D* = 29), layer-id roundtrip;
- integration: full search → fixed point at depth 21 ≤ D*, front = 2,274, certificate verifies 28/28;
- **negative (error-finding) tests**: a forged accuracy, a tampered hardware profile, and a swapped architecture witness are each rejected by specific verifier checks (digest / objective-recompute / state-recompute / SHA-256 input pinning).

## 9. Feasibility, innovation, future work

**Innovation** is not "another multi-objective NAS": it is the **certificate
layer** — optimality claims that a machine (or a judge, in one command) can
refute by tampering, not by trust. **Feasibility:** exact search completes in
~2 s because the DP shares work across the whole space instead of evaluating
architectures one by one. **Honest scope:** guarantees are relative to the
declared search space and device/accuracy models, both SHA-256-pinned.

**Future enhancements:** swap profile tables for on-device measurements
(no code changes — that is the point of the profile format); plug real
trained accuracy instead of the saturation model; extend the skyline with
numpy for 6+ objectives and larger fronts; containerize for deployment;
export certificates as signed artifacts for CI gating.

## 10. Project structure

```
nasmo/         space · models · search · certify · verify · report
web/           index.html          single-file frontend
profiles/      desktop_cpu.json (default) · raspberry_pi4.json (edge comparison)
                hardware-aware cost/fairness tables
config/        space.json          search space + accuracy model
docs/          THEORY.md           formal proofs (cutoff, fixed point, soundness)
certs/         front.json, coverage.json, runs/run_N/…
tests/         test_core.py        23 checks incl. tamper rejection
demo.py        one-command end-to-end pipeline
server.py      FastAPI + SQLite backend
report.html    generated hardware-aware performance report
```
