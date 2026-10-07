"""Objective (proxy) models: exact integer cost model + accuracy model.

State vector of a partial/full architecture (all exact integers):

    (c_eff, latency_us, energy_uj, params_bytes, act_max_bytes, fairness_bp)

where ``c_eff = min(total_capacity, capacity_saturation)``.  All five
reported objectives are monotone functions of the state vector:

    accuracy_bp  = f(c_eff)          [max]
    latency_us   = lat + head        [min]
    energy_uj    = eng + head        [min]
    peak_mem     = params + act_max + head  [min]
    fairness_bp  = fair + head       [min]

Monotonicity of these maps is what makes dominance pruning in state
space sound for the reported objectives, and what lets the certified
depth cutoff (cutoff_lemma) transfer completeness from the finite
enumeration to the unbounded space.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from . import OBJECTIVE_ORDER, SENSE
from .space import LayerType, build_layer_types, validate_arch

State = Tuple[int, int, int, int, int, int]  # (c_eff, lat, eng, params, act, fair)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = ROOT / "profiles" / "desktop_cpu.json"
DEFAULT_SPACE = ROOT / "config" / "space.json"


def load_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_profile(path=DEFAULT_PROFILE) -> dict:
    return load_json(path)


def load_space(path=DEFAULT_SPACE) -> dict:
    return load_json(path)


# ---------------------------------------------------------------- accuracy

def accuracy_bp(capacity: int, am: dict) -> int:
    """Accuracy proxy in basis points: strictly increasing below
    saturation (slope >= 1 bp per capacity unit), constant at acc_max
    at/above saturation."""
    c_sat = am["capacity_saturation"]
    if capacity >= c_sat:
        return am["acc_max_bp"]
    return am["acc_min_bp"] + (am["acc_max_bp"] - am["acc_min_bp"]) * capacity // c_sat


# ------------------------------------------------------- cutoff lemma input

def cutoff_lemma(types: Sequence[LayerType], am: dict) -> dict:
    """Certified depth cutoff D* for the UNBOUNDED space.

    Lemma (proof in docs/THEORY.md): if every layer contributes at least
    a_min > 0 capacity and accuracy saturates at c_sat, then for any
    architecture S (any length) there exists T with len(T) <= D* such that
    T dominates-or-equals S in every objective, where

        D* = ceil((c_sat + a_max) / a_min).
    """
    a_min = min(t.cap for t in types)
    a_max = max(t.cap for t in types)
    c_sat = am["capacity_saturation"]
    if a_min <= 0:
        raise ValueError("cutoff lemma requires a_min > 0 (all layers add capacity)")
    if am["acc_max_bp"] <= am["acc_min_bp"]:
        raise ValueError("accuracy model must be increasing")
    slope = (am["acc_max_bp"] - am["acc_min_bp"]) / c_sat
    if slope < 1.0:
        raise ValueError("accuracy must be strictly increasing per capacity unit "
                         "(slope < 1 bp/unit would break injectivity)")
    d_star = (c_sat + a_max + a_min - 1) // a_min  # ceil((c_sat + a_max) / a_min)
    return {"a_min": a_min, "a_max": a_max, "capacity_saturation": c_sat,
            "accuracy_slope_bp_per_unit": slope, "d_star": d_star}


# ------------------------------------------------------------ state & report

def extend_state(state: State, t: LayerType, c_sat: int) -> State:
    c, lat, eng, params, act, fair = state
    return (min(c + t.cap, c_sat), lat + t.lat, eng + t.eng,
            params + t.params, max(act, t.act), fair + t.fair)


def arch_state(arch: Sequence[str], types_by_id: Dict[str, LayerType],
               c_sat: int) -> State:
    state: State = (0, 0, 0, 0, 0, 0)
    for lid in arch:
        state = extend_state(state, types_by_id[lid], c_sat)
    return state


def reported_objectives(state: State, profile: dict, am: dict) -> Dict[str, int]:
    c, lat, eng, params, act, fair = state
    head = profile["head"]
    return {
        "accuracy_bp": accuracy_bp(c, am),
        "latency_us": lat + head["latency_us"],
        "energy_uj": eng + head["energy_uj"],
        "peak_mem_bytes": params + act + head["params_bytes"] + head["activation_bytes"],
        "fairness_bp": fair + head["fairness_bp"],
    }


def objectives_dominates(a: Dict[str, int], b: Dict[str, int]) -> bool:
    """True if a is at least as good as b in every objective and strictly
    better in at least one."""
    better = False
    for k in OBJECTIVE_ORDER:
        if SENSE[k] == "max":
            if a[k] < b[k]:
                return False
            if a[k] > b[k]:
                better = True
        else:
            if a[k] > b[k]:
                return False
            if a[k] < b[k]:
                better = True
    return better


def dominates(x: State, y: State) -> bool:
    """State-space dominance: x >= y on c_eff, x <= y on all costs,
    strict somewhere."""
    if x[0] < y[0]:
        return False
    if any(x[i] > y[i] for i in (1, 2, 3, 4, 5)):
        return False
    return x != y


# ----------------------------------------------------------------- assembly

class Problem:
    """Fully-loaded search problem: profile + space + types + lemma."""

    def __init__(self, profile: dict, space: dict):
        self.profile = profile
        self.space = space
        self.am = space["accuracy_model"]
        self.types: List[LayerType] = build_layer_types(profile, space)
        self.types_by_id = {t.id: t for t in self.types}
        self.lemma = cutoff_lemma(self.types, self.am)
        self.c_sat = self.am["capacity_saturation"]
        self.objectives = space["objectives"]

    @classmethod
    def load(cls, profile_path=DEFAULT_PROFILE, space_path=DEFAULT_SPACE) -> "Problem":
        return cls(load_profile(profile_path), load_space(space_path))

    def evaluate(self, arch: Sequence[str]) -> dict:
        validate_arch(arch, self.types_by_id)
        state = arch_state(arch, self.types_by_id, self.c_sat)
        return {"state": list(state),
                "objectives": reported_objectives(state, self.profile, self.am),
                "depth": len(arch)}
