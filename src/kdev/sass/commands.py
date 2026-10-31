"""CLI: sass, sass diff. Owner: P5."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

from kdev.build import _runs
from kdev.contract import Effects, Result
from kdev.registry import Arg, Context, Registry

P = "P5"
FX = Effects(gpu="none", writes=("run_dir", "ledger"))


def _short(name: str) -> str:
    from kdev.build.compile import short_name  # noqa: PLC0415

    return short_name(name) if name else "(kernel)"


def _hist_str(h: dict[str, float], n: int = 9, per: bool = False) -> str:
    items = sorted(h.items(), key=lambda kv: -kv[1])[:n]
    return " ".join(f"{k}={v:.2f}" if per else f"{k}={int(v)}" for k, v in items)


def analyse(kernels: list[Any], *, audits: str, top: int, per: float | None, infos: dict, ptxas_text: str
            ) -> dict[str, Any]:
    """Library entry: per-kernel facts, hot loops, findings (JSON-ready)."""
    from kdev.sass.audits import parse_audit_names, run_audits  # noqa: PLC0415
    from kdev.sass.classes import histogram, opcode_counts  # noqa: PLC0415
    from kdev.sass.loops import hot_loops  # noqa: PLC0415

    names, expected = parse_audit_names(audits)
    out = []
    for k in kernels:
        info = infos.get(k.name)
        loops = hot_loops(k, top=max(top, 1), per=per)
        finds = run_audits(k, names, info=info, ptxas_text=ptxas_text, expected=expected)
        body = [i for i in k.instrs if i.base != "NOP"]
        out.append({"kernel": k.name, "short": _short(k.name), "arch": k.arch, "instrs": len(body),
                    "regs": k.regs, "spill_store": info.spill_store if info else None,
                    "spill_load": info.spill_load if info else None, "histogram": histogram(body),
                    "opcodes": dict(list(opcode_counts(body).items())[:40]),
                    "loops": [h.to_dict() for h in loops[:max(top, 1)]],
                    "loop_score": loops[0].score if loops else 0.0,
                    "findings": [f.to_dict() for f in finds]})
    return {"kernels": out}


def _report_lines(data: dict[str, Any], *, top: int, show_loops: bool, per: float | None) -> list[str]:
    ks = data["kernels"]
    lines: list[str] = []
    if len(ks) > 1:
        lines.append(f"{len(ks)} kernels  (name / instrs / regs / spill st,ld / mix)")
        for k in sorted(ks, key=lambda k: -k["loop_score"])[:8]:
            sp = f"{k['spill_store']},{k['spill_load']}" if k["spill_store"] is not None else "-"
            lines.append(f"  {k['short'][:34]:34s} {k['instrs']:>6} {k['regs'] or '-':>4} {sp:>7}  "
                         f"{_hist_str(k['histogram'], 5)}")
        if len(ks) > 8:
            lines.append(f"  ... {len(ks) - 8} more kernels in sass.json (filter with --kernel)")
    focus = sorted(ks, key=lambda k: -k["loop_score"])[: (1 if len(ks) > 1 else 1)]
    for k in focus:
        lines.append(f"kernel {k['short'][:60]}  instrs {k['instrs']}  regs {k['regs'] or '-'}  {k['arch']}")
        lines.append(f"  mix: {_hist_str(k['histogram'])}")
        for h in k["loops"][: (top if show_loops else 1)]:
            lines.append(f"  loop {h['head']}-{h['tail']} depth {h['depth']}{' inner' if h['innermost'] else ''} "
                         f"n={h['n_instr']}: {_hist_str(h['histogram'])}")
            if per and h.get("per_element"):
                lines.append(f"    per element (/{per:g}): {_hist_str(h['per_element'], per=True)}")
            ops = list(h["opcodes"].items())[:10]
            lines.append("    ops: " + " ".join(f"{o}={n}" for o, n in ops))
            if h.get("lines"):
                lines.append("    src: " + ", ".join(h["lines"][:3]))
        if not k["loops"]:
            lines.append("  no loops found (straight-line kernel)")
    finds = [(k["short"], f) for k in ks for f in k["findings"]]
    rank = {"error": 0, "warn": 1, "info": 2}
    finds.sort(key=lambda x: rank.get(x[1]["severity"], 3))
    shown = 0
    for name, f in finds:
        if shown >= 8:
            lines.append(f"  ... {len(finds) - shown} more findings in sass.json")
            break
        lines.append(f"  {f['severity'].upper():5s} {f['audit']}: {name[:24]}: {f['message'][:150]}")
        if f["severity"] == "error" and f["excerpt"]:
            lines.append(f"        {f['excerpt'][0][:110]}")
        shown += 1
    if not finds:
        lines.append("  audits: no findings")
    return lines


def cmd_sass(args: Namespace, ctx: Context) -> Result:
    from kdev.sass.load import load_target  # noqa: PLC0415
    from kdev.sass.parse import by_name  # noqa: PLC0415

    t = load_target(args.target, home=str(_runs.home(ctx)))
    kernels = by_name(t.kernels, args.kernel)
    res = Result(tool="sass")
    res.warnings += t.warnings
    if not kernels:
        res.status, res.verdict = "fail", "no_kernels"
        res.lines = [f"no kernels matched {args.kernel!r}; available: "
                     + ", ".join(_short(k.name) for k in t.kernels[:10])]
        return res
    data = analyse(kernels, audits=args.audit, top=args.top, per=args.per, infos=t.infos, ptxas_text=t.ptxas_text)
    run = _runs.new_run(ctx, "sass", Path(args.target).stem)
    sp = _runs.write_text(run, "sass.txt", t.sass_text)
    jp = _runs.write_json(run, "sass.json", {"schema": "kdev.sass/1", "target": args.target, "kind": t.kind, **data})
    res.files.update({"sass": str(sp), "sass_json": str(jp)})
    errors = sum(1 for k in data["kernels"] for f in k["findings"] if f["severity"] == "error")
    warns = sum(1 for k in data["kernels"] for f in k["findings"] if f["severity"] == "warn")
    res.status = "fail" if errors else "ok"
    res.verdict = "fail" if errors else ("warn" if warns else "pass")
    top_loop = next((k["loops"][0] for k in sorted(data["kernels"], key=lambda k: -k["loop_score"]) if k["loops"]),
                    None)
    res.summary = {"target": args.target, "kind": t.kind, "n_kernels": len(kernels),
                   "errors": errors, "warns": warns,
                   "audits": sorted({f["audit"] for k in data["kernels"] for f in k["findings"]
                                     if f["severity"] != "info"}),
                   "top_loop": {k: top_loop[k] for k in ("kernel", "head", "n_instr", "histogram")} if top_loop
                   else None,
                   # ledger rows (P4 ingest_result reads summary.kernels)
                   "kernels": [{"kernel": k["kernel"], "regs": k["regs"], "sass_instr": k["instrs"],
                                "spill_store": k["spill_store"], "spill_load": k["spill_load"]}
                               for k in data["kernels"][:64]]}
    res.lines = _report_lines(data, top=args.top, show_loops=args.loops, per=args.per)
    if errors:
        first = next(f for k in data["kernels"] for f in k["findings"] if f["severity"] == "error")
        res.next.append(f"kdev show {run.run_id} --section sass_json  (finding: {first['audit']})")
    res.next.append(f"kdev show {run.run_id} --section sass")
    return _runs.finish(run, res)


def cmd_sass_diff(args: Namespace, ctx: Context) -> Result:
    from kdev.sass.diff import diff_kernels  # noqa: PLC0415
    from kdev.sass.load import load_target  # noqa: PLC0415

    ta = load_target(args.a, home=str(_runs.home(ctx)))
    tb = load_target(args.b, home=str(_runs.home(ctx)))
    diffs = diff_kernels(ta.kernels, tb.kernels, args.kernel)
    res = Result(tool="sass diff", verdict="diff")
    res.warnings += ta.warnings + tb.warnings
    run = _runs.new_run(ctx, "sass_diff", f"{Path(args.a).stem}-vs-{Path(args.b).stem}")
    jp = _runs.write_json(run, "diff.json", {"schema": "kdev.sass_diff/1", "a": args.a, "b": args.b,
                                             "kernels": [d.to_dict() for d in diffs]})
    res.files["diff"] = str(jp)
    changed = [d for d in diffs if d.instr_a != d.instr_b or d.class_delta or d.findings_new or d.findings_gone
               or d.regs_a != d.regs_b]
    res.verdict = "different" if changed else "same"
    lines = [f"{len(diffs)} kernel pair(s), {len(changed)} changed  (A={Path(args.a).name}  B={Path(args.b).name})"]
    for d in sorted(changed, key=lambda d: -abs(d.instr_b - d.instr_a))[:8]:
        nm = _short(d.name_b or d.name_a)
        regs = f" regs {d.regs_a}->{d.regs_b}" if d.regs_a != d.regs_b else ""
        lines.append(f"  {nm[:34]:34s} instrs {d.instr_a}->{d.instr_b} ({d.instr_b - d.instr_a:+d}){regs}")
        cd = sorted(d.class_delta.items(), key=lambda kv: -abs(kv[1]))[:6]
        if cd:
            lines.append("    mix: " + " ".join(f"{k}{v:+.0f}" for k, v in cd))
        ld = sorted(((k, v) for k, v in d.loop_delta.items()), key=lambda kv: (kv[0] != "n_instr", -abs(kv[1])))[:6]
        if ld:
            lines.append("    hot loop: " + " ".join(f"{k}{v:+.0f}" for k, v in ld))
        for m in d.findings_new[:2]:
            lines.append(f"    NEW  {m[:120]}")
        for m in d.findings_gone[:2]:
            lines.append(f"    GONE {m[:120]}")
    if len(changed) > 8:
        lines.append(f"  ... {len(changed) - 8} more in diff.json")
    res.lines = lines
    res.summary = {"pairs": len(diffs), "changed": len(changed),
                   "new_findings": sum(len(d.findings_new) for d in diffs),
                   "gone_findings": sum(len(d.findings_gone) for d in diffs)}
    return _runs.finish(run, res)


def register(reg: Registry) -> None:
    reg.command("sass", "hot-loop opcode mix, per-element histogram, audits "
                "(pdl,spills,serialize,repeat,invariant,expect)",
                cmd_sass,
                args=[Arg(("target",), "built submission .py/.cu, .so, .cubin, or SASS text (.sass/.nvd)",
                          type="path"),
                      Arg(("--kernel",), "kernel name regex", default=None),
                      Arg(("--loops",), "show top hot loops", action="store_true"),
                      Arg(("--top",), "number of hot loops", type="int", default=1),
                      Arg(("--per",), "elements per loop iteration (per-element histogram)", type="float",
                          default=None),
                      Arg(("--audit",), "comma list: pdl,spills,serialize,repeat,invariant,expect=HGMMA|LDS, all",
                          default="pdl,spills,serialize,repeat,invariant")],
                effects=FX, package=P)
    reg.command("sass diff", "build-vs-build SASS diff", cmd_sass_diff,
                args=[Arg(("a",), "build A (.py/.cu built, .so, .cubin, .sass)", type="path"),
                      Arg(("b",), "build B", type="path"),
                      Arg(("--kernel",), "kernel name regex", default=None)],
                effects=FX, package=P)
