"""Hot-loop extraction and per-element histograms. Owner: P5.

Hotness ranking (static): innermost depth first, then instr count x weight of contained mma/ldg/lds/fp
work; line-info weight when available. ``--per N`` divides the loop's histogram by N elements per
iteration (the '~15 ops / 8 weights' analysis of LESSONS 1).

Score = sum over body instructions of a class weight (mma 8, tensor_mem 4, ldg/lds/sts/ldl_stl 2, fp/mufu/cvt
1.5, others 1) x 4^(depth-1), x 2 for innermost loops. Static only: a loop that runs once per CTA can rank
above the real hot loop; read the head/tail addresses and source lines.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from kdev.sass.cfg import Block, Loop, build_cfg, find_loops, is_innermost, loop_instrs
from kdev.sass.classes import classify, histogram, opcode_counts
from kdev.sass.parse import Instr, Kernel

_W = {"mma": 8.0, "tensor_mem": 4.0, "ldg": 2.0, "lds": 2.0, "sts": 2.0, "ldl_stl": 2.0, "fp32": 1.5,
      "fp16x2": 1.5, "mufu": 1.5, "cvt": 1.5}


@dataclass
class HotLoop:
    kernel: str
    loop: Loop
    instrs: list[Instr] = field(default_factory=list)
    histogram: dict[str, float] = field(default_factory=dict)
    per_element: dict[str, float] | None = None
    score: float = 0.0
    innermost: bool = True

    def opcodes(self, full: bool = False) -> dict[str, int]:
        return opcode_counts(self.instrs, full=full)

    def source_lines(self, top: int = 5) -> list[tuple[str, int]]:
        """Most frequent source lines of the body (needs -lineinfo / nvdisasm -g)."""
        c = Counter(i.line for i in self.instrs if i.line)
        return c.most_common(top)

    def to_dict(self, max_ops: int = 25) -> dict:
        return {"kernel": self.kernel, "head": hex(self.loop.head), "tail": hex(self.loop.tail),
                "depth": self.loop.depth, "innermost": self.innermost, "n_instr": self.loop.n_instr,
                "score": round(self.score, 1), "histogram": self.histogram, "per_element": self.per_element,
                "opcodes": dict(list(self.opcodes().items())[:max_ops]),
                "lines": [f"{_base(ln)} x{n}" for ln, n in self.source_lines()]}


def _base(line: str) -> str:
    """'/long/path/cuda.cu:730' -> 'cuda.cu:730'."""
    return line.replace("\\", "/").rsplit("/", 1)[-1]


def score_loop(instrs: list[Instr], depth: int, innermost: bool) -> float:
    s = sum(_W.get(classify(i), 1.0) for i in instrs if i.base != "NOP")
    return s * (4.0 ** (depth - 1)) * (2.0 if innermost else 1.0)


def all_loops(kernel: Kernel, cfg: dict[int, Block] | None = None, per: float | None = None) -> list[HotLoop]:
    """Every loop of the kernel as a HotLoop, ranked hottest first."""
    cfg = cfg if cfg is not None else build_cfg(kernel)
    loops = find_loops(kernel, cfg)
    out: list[HotLoop] = []
    for lp in loops:
        body = loop_instrs(kernel, cfg, lp)
        inner = is_innermost(lp, loops)
        h = HotLoop(kernel=kernel.name, loop=lp, instrs=body, histogram=histogram(body),
                    per_element=histogram(body, per) if per else None, score=score_loop(body, lp.depth, inner),
                    innermost=inner)
        out.append(h)
    out.sort(key=lambda h: (-h.score, h.loop.head))
    return out


def hot_loops(kernel: Kernel, top: int = 3, per: float | None = None) -> list[HotLoop]:
    return all_loops(kernel, per=per)[: max(0, top)]
