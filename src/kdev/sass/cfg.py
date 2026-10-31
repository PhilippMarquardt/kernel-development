"""Basic blocks, CFG and loops from plain SASS (or nvdisasm -cfg when available). Owner: P5.

Leaders: the first instruction, every branch target, and the instruction after any BRA/BRX/JMP/JMX/EXIT/RET/
CALL/KILL. Successors: unpredicated BRA/JMP -> target; predicated (or ``BRA Px, tgt`` / ``.DIV``) -> target +
fall-through; unpredicated EXIT/RET/KILL -> none; BRX/JMX (indirect) -> none (marked); CALL -> fall-through.
BSSY/BSYNC are reconvergence markers, not edges. Blocks unreachable from the entry (the trailing
``BRA self`` trap after EXIT, padding) are kept but never form loops.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from kdev.sass.parse import Instr, Kernel


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


_ENDS = frozenset({"BRA", "BRX", "JMP", "JMX", "EXIT", "RET", "CALL", "KILL"})
_UNCOND_STOP = frozenset({"EXIT", "RET", "KILL"})


def _is_cond(ins: Instr) -> bool:
    if ins.pred and ins.pred not in ("@PT",):
        return True
    if ".DIV" in ins.opcode or ".CONV" in ins.opcode:
        return True
    # 'BRA P6, 0xf20' / 'BRA !P0, tgt': the branch carries its own condition
    return len(ins.operands) >= 2 and ins.operands[0].lstrip("!").startswith(("P", "UP")) and \
        ins.operands[0].lstrip("!") not in ("PT", "UPT")


def build_cfg(kernel: Kernel) -> dict[int, Block]:
    ins = kernel.instrs
    if not ins:
        return {}
    addrs = [i.addr for i in ins]
    aset = set(addrs)
    leaders = {addrs[0]}
    for k, i in enumerate(ins):
        if i.base in _ENDS:
            if i.target is not None and i.target in aset and i.base in ("BRA", "JMP", "CALL"):
                leaders.add(i.target)
            if k + 1 < len(ins):
                leaders.add(addrs[k + 1])
    blocks: dict[int, Block] = {}
    order = sorted(leaders)
    idx = {a: k for k, a in enumerate(addrs)}
    for n, start in enumerate(order):
        end_k = (idx[order[n + 1]] - 1) if n + 1 < len(order) else len(ins) - 1
        blocks[start] = Block(start=start, end=addrs[end_k])
    for start, b in blocks.items():
        last = ins[idx[b.end]]
        nxt = addrs[idx[b.end] + 1] if idx[b.end] + 1 < len(ins) else None
        succs: list[int] = []
        if last.base in ("BRA", "JMP"):
            if last.target is not None and last.target in blocks:
                succs.append(last.target)
            if _is_cond(last) and nxt is not None:
                succs.append(nxt)
        elif last.base in _UNCOND_STOP:
            if last.pred and nxt is not None:
                succs.append(nxt)
        elif last.base in ("BRX", "JMX"):
            if last.pred and nxt is not None:
                succs.append(nxt)
        elif nxt is not None:   # CALL and plain fall-through
            succs.append(nxt)
        b.succs = list(dict.fromkeys(succs))
    for start, b in blocks.items():
        for s in b.succs:
            blocks[s].preds.append(start)
    return blocks


def reachable(cfg: dict[int, Block], entry: int | None = None) -> set[int]:
    if not cfg:
        return set()
    entry = min(cfg) if entry is None else entry
    seen, stack = set(), [entry]
    while stack:
        b = stack.pop()
        if b in seen:
            continue
        seen.add(b)
        stack.extend(cfg[b].succs)
    return seen


def dominators(cfg: dict[int, Block]) -> dict[int, set[int]]:
    """Iterative dominator sets over the blocks reachable from the entry."""
    live = reachable(cfg)
    if not live:
        return {}
    entry = min(cfg)
    order = sorted(live)
    dom = {b: set(live) for b in order}
    dom[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for b in order:
            if b == entry:
                continue
            preds = [p for p in cfg[b].preds if p in live]
            new = set.intersection(*(dom[p] for p in preds)) if preds else set()
            new = new | {b}
            if new != dom[b]:
                dom[b] = new
                changed = True
    return dom


def block_instrs(kernel: Kernel, cfg: dict[int, Block]) -> dict[int, list[Instr]]:
    """{block start: its instructions in address order}."""
    find = block_index(cfg)
    out: dict[int, list[Instr]] = {b: [] for b in cfg}
    for i in kernel.instrs:
        b = find(i.addr)
        if b is not None:
            out[b].append(i)
    return out


def find_loops(kernel: Kernel, cfg: dict[int, Block] | None = None) -> list[Loop]:
    """Natural loops from back edges (branch to an address <= its own that dominates it). Back edges to a
    non-dominating header (irreducible flow) fall back to the address range [target, branch]. Loops with
    the same header are merged. Sorted by header address; ``depth`` = nesting level (1 = outermost)."""
    cfg = cfg if cfg is not None else build_cfg(kernel)
    if not cfg:
        return []
    live = reachable(cfg)
    dom = dominators(cfg)
    loops: dict[int, Loop] = {}
    for b in sorted(live):
        blk = cfg[b]
        for s in blk.succs:
            if s > blk.end or s not in live:
                continue   # forward edge
            if s in dom.get(b, set()):
                body = {s, b}
                stack = [b] if b != s else []
                while stack:
                    x = stack.pop()
                    for p in cfg[x].preds:
                        if p in live and p not in body:
                            body.add(p)
                            stack.append(p)
            else:
                body = {x for x in live if s <= x <= blk.end}
            lp = loops.get(s)
            if lp is None:
                loops[s] = Loop(head=s, tail=blk.end, blocks=sorted(body))
            else:
                lp.blocks = sorted(set(lp.blocks) | body)
                lp.tail = max(lp.tail, blk.end)
    out = sorted(loops.values(), key=lambda lp: lp.head)
    bi = block_instrs(kernel, cfg)
    for lp in out:
        lp.n_instr = sum(1 for b in lp.blocks for i in bi[b] if i.base != "NOP")
        s = set(lp.blocks)
        lp.depth = 1 + sum(1 for o in out if o is not lp and s < set(o.blocks))
    return out


def loop_instrs(kernel: Kernel, cfg: dict[int, Block], loop: Loop) -> list[Instr]:
    """Instructions of a loop body in address order."""
    bi = block_instrs(kernel, cfg)
    return sorted((i for b in loop.blocks for i in bi.get(b, [])), key=lambda i: i.addr)


def is_innermost(loop: Loop, loops: list[Loop]) -> bool:
    s = set(loop.blocks)
    return not any(o is not loop and set(o.blocks) < s for o in loops)


def _block_of(cfg: dict[int, Block], addr: int) -> int | None:
    return block_index(cfg)(addr)


def reachable_avoiding(kernel: Kernel, cfg: dict[int, Block], stops: set[int]) -> set[int]:
    """Instruction addresses reachable from the entry on paths that do not execute any address in ``stops``
    (the stop instructions themselves are excluded)."""
    if not cfg:
        return set()
    by_block = {b: [i.addr for i in v] for b, v in block_instrs(kernel, cfg).items()}
    seen_blocks: set[int] = set()
    out: set[int] = set()
    stack = [min(cfg)]
    while stack:
        b = stack.pop()
        if b in seen_blocks:
            continue
        seen_blocks.add(b)
        blocked = False
        for a in by_block[b]:
            if a in stops:
                blocked = True
                break
            out.add(a)
        if not blocked:
            stack.extend(cfg[b].succs)
    return out


def can_reach(kernel: Kernel, cfg: dict[int, Block], targets: set[int]) -> set[int]:
    """Instruction addresses from which some address in ``targets`` is reachable (reverse search over
    blocks; inside a target's own block only the instructions before it, unless a cycle leads back)."""
    tblocks: dict[int, int] = {}
    for t in targets:
        b = _block_of(cfg, t)
        if b is not None:
            tblocks[b] = min(t, tblocks.get(b, t))
    back: set[int] = set()            # blocks with a path of >= 1 edge into a target block
    stack = [p for b in tblocks for p in cfg[b].preds]
    while stack:
        b = stack.pop()
        if b in back:
            continue
        back.add(b)
        stack.extend(cfg[b].preds)
    out: set[int] = set()
    find = block_index(cfg)
    for i in kernel.instrs:
        b = find(i.addr)
        if b in back or (b in tblocks and i.addr < tblocks[b]):
            out.add(i.addr)
    return out


def block_index(cfg: dict[int, Block]):
    """addr -> block start lookup (bisect over block starts)."""
    import bisect  # noqa: PLC0415

    starts = sorted(cfg)

    def find(addr: int) -> int | None:
        k = bisect.bisect_right(starts, addr) - 1
        if k < 0:
            return None
        b = starts[k]
        return b if addr <= cfg[b].end else None

    return find


def reachable_before(kernel: Kernel, cfg: dict[int, Block], addr: int) -> set[int]:
    """Instruction addresses on some path from entry to ``addr`` (used by the PDL audit): reachable from the
    entry without passing ``addr`` and able to reach ``addr``."""
    return reachable_avoiding(kernel, cfg, {addr}) & can_reach(kernel, cfg, {addr})
