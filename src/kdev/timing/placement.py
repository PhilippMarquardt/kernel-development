"""Placement logging (judge mode): where the buffers a timed call touches sit in the address space.

Placement-sensitive shapes move by up to ~1 us with allocation history (PROTOCOL 6). For every label of a
judge-mode ``time`` process we record, for the largest tensors reachable from its prepare state (weights)
and for the captured outputs of each shape, ``data_ptr % 2 MiB`` and ``% 64 KiB``. The placement *class*
of a tensor is ``A`` (2 MiB aligned), ``B`` (64 KiB aligned) or ``C`` (neither); a label's class is the
string over its top-K tensors in a stable order. ``ab`` compares the classes of the compared files'
processes and warns when they differ (the A/B delta may then be placement, not code).
"""
from __future__ import annotations

import types
from typing import Any

TOP = 8
MB2 = 2 << 20
KB64 = 64 << 10


def _is_tensor(x: Any) -> bool:
    return hasattr(x, "data_ptr") and hasattr(x, "element_size") and hasattr(x, "numel") and hasattr(x, "is_cuda")


def tensors_in(obj: Any, prefix: str = "", depth: int = 4, seen: set[int] | None = None) -> list[tuple[str, Any]]:
    """(name, tensor) pairs of CUDA tensors reachable from obj (dict/list/tuple/nn.Module/object attributes)."""
    seen = set() if seen is None else seen
    out: list[tuple[str, Any]] = []
    if obj is None or id(obj) in seen or depth < 0:
        return out
    seen.add(id(obj))
    if _is_tensor(obj):
        if getattr(obj, "is_cuda", False) and obj.numel():
            out.append((prefix or "t", obj))
        return out
    if isinstance(obj, (str, bytes, int, float, bool, type)) or callable(obj) and not hasattr(obj, "__dict__"):
        return out
    if isinstance(obj, types.ModuleType) and prefix:          # never walk into imported modules (torch, ...)
        return out
    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, (list, tuple)):
        items = list(enumerate(obj))
    elif hasattr(obj, "named_parameters") and hasattr(obj, "named_buffers"):
        try:
            items = list(obj.named_parameters()) + list(obj.named_buffers())
        except Exception:  # noqa: BLE001
            items = []
    elif isinstance(obj, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
        return out
    elif hasattr(obj, "__dict__"):
        items = [(k, v) for k, v in vars(obj).items() if not k.startswith("__")]
    else:
        return out
    for k, v in items[:4096]:
        out += tensors_in(v, f"{prefix}.{k}" if prefix else str(k), depth - 1, seen)
    return out


def klass(ptr: int) -> str:
    return "A" if ptr % MB2 == 0 else ("B" if ptr % KB64 == 0 else "C")


def describe(named: list[tuple[str, Any]], top: int = TOP) -> dict[str, Any]:
    """Top-``top`` tensors by size (ties by name): name, MiB, ptr % 2 MiB, ptr % 64 KiB; plus the class string."""
    uniq: dict[int, tuple[str, Any]] = {}
    for n, t in named:
        try:
            p = int(t.data_ptr())
        except Exception:  # noqa: BLE001
            continue
        uniq.setdefault(p, (n, t))
    rows = []
    for p, (n, t) in uniq.items():
        rows.append({"name": n[:80], "mib": round(t.numel() * t.element_size() / 2**20, 3), "ptr_mod_2m": p % MB2,
                     "ptr_mod_64k": p % KB64, "cls": klass(p)})
    rows.sort(key=lambda r: (-r["mib"], r["name"]))
    rows = rows[:top]
    return {"tensors": rows, "class": "".join(r["cls"] for r in rows), "n_total": len(uniq)}


def compare(per_proc: list[tuple[str, dict[str, Any]]], labels: list[str]) -> list[str]:
    """Warnings when the processes of two compared labels differ in placement class (weights or outputs)."""
    by: dict[str, dict[str, set[str]]] = {}
    for lb, pl in per_proc:
        if not pl:
            continue
        d = by.setdefault(lb, {"weights": set(), "outputs": set()})
        d["weights"].add(str(pl.get("weights_class", "")))
        d["outputs"].add(str(pl.get("outputs_class", "")))
    warns = []
    ref = labels[0] if labels else None
    if ref not in by:
        return warns
    for lb in labels[1:]:
        if lb not in by:
            continue
        for part in ("outputs", "weights"):
            a, b = by[ref][part], by[lb][part]
            if a and b and a != b:
                warns.append(f"PLACEMENT: {part} class differs {ref} {'/'.join(sorted(a))} vs {lb} "
                             f"{'/'.join(sorted(b))} (data_ptr %2MB: A aligned, B 64KB, C other): the delta may be "
                             "placement, not code; re-run with --pad-mb N (same pads) or more --rounds")
    return warns
