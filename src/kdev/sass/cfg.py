"""Basic blocks, CFG and loops from plain SASS (or nvdisasm -cfg when available). Owner: P5."""
from __future__ import annotations

from dataclasses import dataclass, field

from kdev.sass.parse import Kernel


@dataclass
class Block:
    start: int
    end: int                       # address of the last instruction
    succs: list[int] = field(default_factory=list)
    preds: list[int] = field(default_factory=list)


@dataclass
class Loop:
    head: int                      # header block start address
    tail: int                      # address of the back-edge branch
    blocks: list[int] = field(default_factory=list)
    depth: int = 1
    n_instr: int = 0


def build_cfg(kernel: Kernel) -> dict[int, Block]:
    raise NotImplementedError("cfg.build_cfg (P5)")


def find_loops(kernel: Kernel, cfg: dict[int, Block] | None = None) -> list[Loop]:
    """Natural loops from back edges (branch to an address <= its own that dominates it)."""
    raise NotImplementedError("cfg.find_loops (P5)")


def reachable_before(kernel: Kernel, cfg: dict[int, Block], addr: int) -> set[int]:
    """Instruction addresses on some path from entry to ``addr`` (used by the PDL audit)."""
    raise NotImplementedError("cfg.reachable_before (P5)")
