# NASMO — Theory of Global Optimality

This document gives the formal arguments behind NASMO's claim: **the returned
Pareto front is the complete, globally optimal set of non-dominated
architectures for any combination of the five objectives, over a search space
of unbounded size** — together with a machine-checkable certificate
(`certs/front.json` + `certs/coverage.json`) and an independent verifier
(`nasmo/verify.py`) that re-derives every claim from the raw inputs.

## 1. Setting

**Search space.** An architecture is a sequence `S = (l_1, …, l_n)` of
layer types, `n ≥ 1`, with **no upper bound on n**. Each layer type is a
triple (op, width, residual); the space contains 36 layer types for the
default configuration and grows only with the declared op/width/residual
lists in `config/space.json`.

**State vector.** Every architecture maps to an exact-integer state

    state(S) = (c_eff,  lat,  eng,  params,  act_max,  fair)

computed by the recurrences in `nasmo/models.py`:

    c_eff  = min(Σ capacity, c_sat)      (saturating capacity)
    lat    = Σ per-layer latency           (strictly additive, > 0 per layer)
    eng    = Σ per-layer energy            (strictly additive, > 0 per layer)
    params = Σ per-layer parameter bytes   (additive, ≥ 0)
    act_max = max per-layer activation     (idempotent max)
    fair   = Σ per-layer fairness gap      (additive, ≥ 0)

**Reported objectives.** The five deliverable objectives are exact-integer
functions of the state plus a constant device head:

| objective        | sense | formula                          |
|------------------|-------|----------------------------------|
| `accuracy_bp`    | max   | `f(c_eff)` — strictly increasing below `c_sat`, constant `acc_max` above |
| `latency_us`     | min   | `lat + head_lat`                 |
| `energy_uj`      | min   | `eng + head_eng`                 |
| `peak_mem_bytes` | min   | `params + act_max + head`        |
| `fairness_bp`    | min   | `fair + head_fair`               |

**Dominance.** `x ⪯ y` ("x dominates y") iff x is at least as good as y in
all five reported objectives and strictly better in at least one. The
*Pareto front* is the set of architectures not dominated by any architecture
in the space. "Globally optimal for any objective combination" means: the
front is exactly this set, so any scalarization (weights, thresholds,
lexicographic preference) over the five objectives has its optimum on the
returned front.

## 2. Lemma 1 — State dominance is sound (monotone reporting)

**Claim.** If `state(x) ⪯_state state(y)` (c_eff ≥, every cost ≤, strict
somewhere), then the reported objectives of x dominate-or-equal those of y.

**Proof.** Each reported objective is a coordinate-wise *monotone* function
of the state: `f` is non-decreasing in `c_eff`; `lat+head`, `eng+head`,
`fair+head` are non-decreasing in their coordinate (identity + constant);
`params + act_max + head` is non-decreasing in both params and act_max.
A monotone map preserves the "at least as good" direction on every
coordinate, so all five objectives of x are ≥ (accuracy) / ≤ (costs) those
of y. Strictness in any state coordinate that appears with a *positive
slope* in some objective gives strictness there — and every state
coordinate has positive slope in at least one reported objective
(c_eff → accuracy below saturation; lat, eng, params, act_max, fair →
their cost). The only flattening is `f` at saturation, where equality is
harmless: dominance may collapse to equality, which the tie-break resolves
deterministically (shorter, then lexicographically smallest). ∎

**Consequence.** Pruning in state space never discards a reported-objective
nondominated architecture. This is what licenses the exact skyline over
six-integer states instead of five-objective vectors.

## 3. Lemma 2 — Certified depth cutoff for the unbounded space

Let `a_min = min capacity over layer types` (> 0: every layer adds
capacity — machine-checked in `cutoff_lemma`), `a_max = max capacity`, and
`c_sat` the accuracy saturation point. Define

    D* = ceil((c_sat + a_max) / a_min).

**Claim (cutoff lemma).** For every architecture S (of *any* length) there
exists an architecture T with `len(T) ≤ D*` such that T dominates-or-equals
S in every reported objective. Hence the front of the unbounded space equals
the front over depths `1..D*`.

**Proof.** Two cases.

*Case 1 — `len(S) ≤ D*`: take T = S.*

*Case 2 — `len(S) > D*`: saturation kicks in.* Every layer contributes
`≥ a_min` capacity, so

    cap(S) ≥ len(S) · a_min ≥ (D* + 1) · a_min > c_sat + a_max ≥ c_sat,

so S is saturated: `accuracy(S) = acc_max`. Let P be the **shortest prefix**
of S whose accumulated capacity reaches `c_sat`. Since a prefix of k layers
has capacity `≥ k · a_min`, saturation is reached by `k = ceil(c_sat/a_min)`
layers, so

    len(P) ≤ ceil(c_sat / a_min) ≤ ceil((c_sat + a_max)/a_min) = D*.

P uses a subset of S's layers in the same order, so every additive cost of
P (latency, energy, params, fairness) is ≤ S's, and its `act_max` is ≤
S's (`max` over fewer terms). Both are saturated, so accuracy is equal
(`acc_max`). Hence P dominates-or-equals S. ∎

**Machine-check.** The verifier re-derives `a_min, a_max, c_sat, D*` from
the raw `profiles/*.json` + `config/space.json` and compares them
bit-for-bit with the certificate (`cutoff lemma re-derived identically`),
and re-runs the search to depth `D*` (`re-run:` checks).

## 4. Lemma 3 — Pareto dynamic programming is exact up to depth D*

The search (`nasmo/search.py`) never enumerates the space; it keeps, per
depth d, only the skyline `G_d` of all generated states and extends **only**
frontier states:

    C_1 = all single layers
    C_d = { extend(s, l) : s ∈ F_{d-1}, l ∈ layer_types }   (F = exact-depth frontier states)
    G_d = skyline(G_{d-1} ∪ C_d)
    F_d = C_d ∩ G_d

**Claim A (extension soundness).** If `state(t) ⪯ state(p)` then
`state(t·l) ⪯ state(p·l)` for every layer l.

*Proof.* Capacity: `min(cap + Δ, c_sat)` is monotone in `cap`. Additive
costs: adding the same non-negative increment to a ≤ relation preserves it.
`act_max`: `max(a, Δ)` is monotone in a. Strictness: if t is strictly
better than p in any *additive* coordinate, `t·l` stays strictly better
there (the increments cancel); if strictness is only in capacity or only in
`act_max`, saturation/max may collapse it — in that case `t·l` and `p·l`
are equal states and the deterministic tie-break keeps one representative.
Either way no nondominated extension is lost. ∎

**Claim B (frontier-only extension is complete).** Every architecture S of
depth `d ≤ D*` has its state dominated-or-equalled by some state in `G_d`.

*Proof sketch by induction on depth.* Depth 1: `C_1` contains every
single-layer state; the skyline either keeps it or a kept point dominates
it. Inductive step: S = p·l. By IH some `t ∈ G_{d-1}` dominates-or-equals
p. If t was born at depth d−1 (i.e. t ∈ F_{d−1}) Claim A gives
`t·l ⪯ p·l = S` and `t·l ∈ C_d`, so `G_d` (which dominates-or-keeps all of
`C_d`) covers S. If t was inherited from earlier rounds, t was already
extended by every layer at its birth round; re-running the same argument on
the earliest point that dominates the dropped candidate yields a dominated
chain that terminates on a frontier point already in `G_d` (chains are
strict and the generated state set is finite). Duplicate vectors are
collapsed to the deterministically smallest architecture, so no state is
ever represented twice. ∎

**Corollary.** `G_{D*}` = the exact Pareto front over depths `1..D*`, and
by Lemma 2 also over the **unbounded** space.

## 5. Lemma 4 — Fixed-point stopping is safe

The search may stop before `D*` when `G_d == G_{d-1}` (same key set).

**Claim.** If `G_d == G_{d-1}` then `G_{d+1} == G_d`, and therefore
`G_d = G_{D*}`: the frontier can never change again.

*Proof.* Let `c ∈ C_{d+1}`. Then `c = extend(f, l)` for `f ∈ F_d ⊆ G_d`,
and f was born at some depth k ≤ d−1, so `c` was already generated no later
than round k+1 ≤ d. If c was on the frontier at its birth it either stayed
on every subsequent frontier — hence is already in `G_d = G_{d-1}` — or was
dropped by a strict dominator, and iterating "dropped by a dominator that
itself was later dropped" gives a strictly decreasing domination chain over
the finite generated state set, which can only terminate at a point still
on the frontier at round d, i.e. in `G_d`. Either way every point of
`C_{d+1}` is dominated-or-equalled by a point of `G_d`, so
`skyline(G_d ∪ C_{d+1}) = G_d`. Induction carries this to every depth up
to `D*`. ∎

**Machine-check.** `coverage.json` records, for every round, the frontier
size and a SHA-256 of the sorted state set, plus which round first reported
`fixed_point`. The verifier checks the round sequence, the stop depth, and
that a full re-run reproduces **every round hash** (`re-run: identical
per-round frontier hashes`).

## 6. What the certificate contains, and what it does not

The certificate proves: *within the declared search space
(`config/space.json`) and the declared cost/accuracy models
(`profiles/<device>.json`), no architecture of any length exists whose
reported five-tuple dominates any returned front point.* It is relative to
those declared models — exactly what a machine-checkable certificate of
optimality for every architecture decision means in practice. The profile tables are illustrative microbenchmarks for
the demo; replacing them with on-device measurements changes no code and
invalidates old certificates by hash (the verifier re-hashes both inputs).

**Checkable in one command:**

    python -m nasmo.verify certs

which re-derives the lemma, the layer table, every state and objective,
mutual non-domination of the front, the coverage log, and the entire search
frontier from raw inputs — 28 independent checks, `CERTIFICATE VALID` iff
all pass.
