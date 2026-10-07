"""Exact multi-objective search: Pareto dynamic programming over the
unbounded layer-sequence space.

Algorithm (proofs in docs/THEORY.md):

    G_0 = {}
    for depth d = 1 .. D*:
        C_d = { extend(s, l) : s in F_{d-1}, l in layer_types }  # exact-depth candidates
        H_d = C_d merged with G_{d-1}   # duplicate vectors collapsed
                                       # deterministically to the smallest arch
        G_d = nondominated(H_d)        # exact skyline
        F_d = { s in G_d : s came from C_d }
        if G_d == G_{d-1}: STOP        # fixed point: frontier can never change again
    return G_stop, coverage log

Two stopping rules, both machine-checked by the verifier:

* ``fixed_point``  - G_d == G_{d-1}.  THEORY.md proves the frontier is
  then complete for sequences of *every* length (unbounded space).
* ``cutoff_bound`` - depth reached D* = ceil((c_sat + a_max)/a_min).
  The cutoff lemma then guarantees completeness for the unbounded space.

Because all quantities are exact integers, every dominance test is
bit-exact.  The skyline sweeps candidates in an order where every
dominator precedes its dominated (costs ascending, c_eff descending) and
groups accepted states by activation footprint (a "maximize" dimension),
so a candidate is rejected after comparing only against per-group
aggregate bounds in the common case.  Latency needs no check: insertion
order already guarantees every accepted state has latency <= the
candidate's.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

from .models import Problem, State
from .space import LayerType

Front = Dict[State, Tuple[str, ...]]  # state -> arch


def _vec_hash(states) -> str:
    h = hashlib.sha256()
    for s in sorted(states):
        h.update(("%d,%d,%d,%d,%d,%d\n" % s).encode())
    return h.hexdigest()


def _better_arch(a: Tuple[str, ...], b: Tuple[str, ...]) -> bool:
    """Deterministic tie-break for identical vectors: shallower first,
    then lexicographic."""
    return (len(a), a) < (len(b), b)


class _Group:
    """States sharing one activation footprint (act is a 'min' objective,
    so a candidate only needs to face groups with act <= its own)."""
    __slots__ = ("states", "max_c", "min_eng", "min_params", "min_fair")

    def __init__(self) -> None:
        self.states: List[State] = []
        self.max_c = -1
        self.min_eng = 1 << 62
        self.min_params = 1 << 62
        self.min_fair = 1 << 62

    def offer(self, s: State) -> None:
        self.states.append(s)
        if s[0] > self.max_c:
            self.max_c = s[0]
        if s[2] < self.min_eng:
            self.min_eng = s[2]
        if s[3] < self.min_params:
            self.min_params = s[3]
        if s[5] < self.min_fair:
            self.min_fair = s[5]


def skyline(items: Dict[State, Tuple[str, ...]]) -> Dict[State, Tuple[str, ...]]:
    """Exact skyline: keep only nondominated states."""
    order = sorted(items, key=lambda s: (s[1], s[2], s[3], s[4], s[5], -s[0]))
    groups: Dict[int, _Group] = {}
    kept: Dict[State, Tuple[str, ...]] = {}
    for s in order:
        s0, s1, s2, s3, s4, s5 = s
        dominated = False
        for act_g, g in groups.items():
            if act_g > s4:
                continue
            # necessary conditions for a dominator inside this group
            if (g.max_c >= s0 and g.min_eng <= s2
                    and g.min_params <= s3 and g.min_fair <= s5):
                for t in g.states:
                    if (t[0] >= s0 and t[2] <= s2
                            and t[3] <= s3 and t[5] <= s5):
                        dominated = True
                        break
                if dominated:
                    break
        if not dominated:
            g = groups.get(s4)
            if g is None:
                g = groups[s4] = _Group()
            g.offer(s)
            kept[s] = items[s]
    return kept


def run_search(prob: Problem) -> Tuple[Front, List[dict], dict]:
    """Run the certified Pareto DP.

    Returns (frontier, per-round coverage log, stop metadata).
    """
    types: List[LayerType] = prob.types
    c_sat = prob.c_sat
    d_star = prob.lemma["d_star"]

    # precomputed extension table: (cap, lat, eng, params, act, fair) per type
    tl = [(t.cap, t.lat, t.eng, t.params, t.act, t.fair, t.id) for t in types]
    zero: State = (0, 0, 0, 0, 0, 0)

    G: Front = {}
    F: Front = {}
    rounds: List[dict] = []
    stop_reason = "cutoff_bound"
    stop_depth = d_star

    for depth in range(1, d_star + 1):
        C: Front = {}
        if depth == 1:
            for cap, lat, eng, params, act, fair, tid in tl:
                s = (cap, lat, eng, params, act, fair)
                prev = C.get(s)
                arch = (tid,)
                if prev is None or _better_arch(arch, prev):
                    C[s] = arch
        else:
            for (c0, l0, e0, p0, a0, f0), base_arch in F.items():
                if c0 == c_sat:
                    # Saturated parent: the extension keeps c_eff but adds
                    # strictly positive latency, so it is dominated by its
                    # own parent and can never be on the front.
                    continue
                for cap, lat, eng, params, act, fair, tid in tl:
                    s = (min(c0 + cap, c_sat), l0 + lat, e0 + eng,
                         p0 + params, a0 if a0 > act else act, f0 + fair)
                    prev = C.get(s)
                    if prev is None:
                        C[s] = base_arch + (tid,)
                    else:
                        arch = base_arch + (tid,)
                        if (len(arch), arch) < (len(prev), prev):
                            C[s] = arch

        H: Front = dict(G)
        if C:
            if len(C) <= len(H):
                for s, arch in C.items():
                    prev = H.get(s)
                    if prev is None or (len(arch), arch) < (len(prev), prev):
                        H[s] = arch
            else:
                H2 = dict(C)
                for s, arch in H.items():
                    prev = H2.get(s)
                    if prev is None or (len(arch), arch) < (len(prev), prev):
                        H2[s] = arch
                H = H2

        G_new = skyline(H)
        F = {s: arch for s, arch in C.items() if s in G_new}
        fixed = (len(G_new) == len(G) and G_new.keys() == G.keys())
        rounds.append({"depth": depth, "candidates": len(C),
                       "merged": len(H), "frontier": len(G_new),
                       "hash": _vec_hash(G_new.keys()),
                       "fixed_point": fixed})
        G = G_new
        if fixed and depth >= 2:
            stop_reason = "fixed_point"
            stop_depth = depth
            break

    meta = {"stop_reason": stop_reason, "stop_depth": stop_depth,
            "d_star": d_star, "front_size": len(G)}
    return G, rounds, meta
