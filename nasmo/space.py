"""Search-space encoding: architectures are sequences of layer ids.

A layer id looks like ``sepconv3@1.0R`` (op, width, residual flag).
An architecture is a non-empty tuple of layer ids; sequence length is
unbounded (see models.cutoff_lemma for the certified finite cutoff).
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, NamedTuple, Sequence, Tuple

LAYER_RE = re.compile(r"^([a-z][a-z0-9]*)@([0-9]+(?:\.[0-9]+)?)([RN])$")


class LayerType(NamedTuple):
    id: str
    op: str
    width: float
    residual: bool
    cap: int      # capacity units contributed by this layer
    lat: int      # us
    eng: int      # uJ
    params: int   # bytes
    act: int      # bytes (activation footprint)
    fair: int     # fairness gap basis points


def make_layer_id(op: str, width: float, residual: bool) -> str:
    return f"{op}@{width}{'R' if residual else 'N'}"


def parse_layer_id(layer_id: str) -> Tuple[str, float, bool]:
    m = LAYER_RE.match(layer_id)
    if not m:
        raise ValueError(f"malformed layer id: {layer_id!r}")
    return m.group(1), float(m.group(2)), m.group(3) == "R"


def build_layer_types(profile: dict, space: dict) -> List[LayerType]:
    """Enumerate every layer type in the space, deterministically sorted."""
    ops = profile["ops"]
    residual_cfg = profile["residual"]
    types: List[LayerType] = []
    for op in space["ops"]:
        if op not in ops:
            raise ValueError(f"space references unknown op {op!r}")
        t = ops[op]
        for width in space["widths"]:
            w2 = width * width
            for residual in space["residual_flags"]:
                cap = int(round(t["capacity_base"] * width))
                lat = int(round(t["latency_us"] * width))
                eng = int(round(t["energy_uj"] * width))
                params = int(round(t["params_bytes"] * w2))
                act = int(round(t["activation_bytes"] * w2))
                fair = int(round(t["fairness_bp"] * width))
                if residual:
                    cap += residual_cfg["extra_capacity"]
                    lat += residual_cfg["extra_latency_us"]
                    eng += residual_cfg["extra_energy_uj"]
                    params += residual_cfg["extra_params_bytes"]
                    fair += residual_cfg["extra_fairness_bp"]
                lid = make_layer_id(op, width, residual)
                if lat <= 0 or eng <= 0:
                    raise ValueError(f"layer {lid} must have positive latency/energy")
                types.append(LayerType(lid, op, width, residual, cap, lat, eng,
                                       params, act, fair))
    types.sort(key=lambda t: t.id)
    ids = [t.id for t in types]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate layer ids in space")
    return types


def validate_arch(arch: Sequence[str], types_by_id: dict) -> None:
    if not isinstance(arch, (list, tuple)) or len(arch) < 1:
        raise ValueError("architecture must be a non-empty list of layer ids")
    for lid in arch:
        if lid not in types_by_id:
            raise ValueError(f"layer {lid!r} not in search space")


def arch_digest(arch: Iterable[str]) -> str:
    h = hashlib.sha256("|".join(arch).encode()).hexdigest()
    return h[:12]
