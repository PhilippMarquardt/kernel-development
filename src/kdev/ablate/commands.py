"""CLI: ablate. ``kdev ablate FILE --component x_loads,mma,... [--variant C=PATH] [--set C:K=V]``.

Variants (timing only, WRONG RESULTS by design) per component, first match wins:
  1. ``--variant COMPONENT=PATH``: a ready ablation file;
  2. ``--set COMPONENT:K=V`` (repeatable): the base file with knobs/defines (compiled in as -DK=V by the
     kdev build cache, also $KDEV_KNOBS), e.g. ``--set mma:XS_NO_MMA=1`` when the file has the hooks;
  3. the per-problem hook ``kdev_ablate.py`` (``--hook PATH``, else next to FILE or 1-2 dirs up):
     ``def ablate(base: str, component: str, out_dir: str) -> str | dict | None`` returning a file path or
     ``{"path": ..., "knobs": {...}}`` (source transforms stay per problem; see xs/mk2.py of muse_bf16_proj).
``--control PATH`` (or a hook result for ``control``) = all hooks compiled in, nothing removed: noise band.
Then: build every variant (kdev build, with discovery), one judge-mode ``ab`` (base = A first, variants after),
and the budget table (kdev.ablate.budget) printed + ``budget.md``/``budget.json`` in the ab run dir.
``--from-run RUN`` re-renders a finished ab (labels from its ablate.json, or ``--map B=x_loads,C=mma``).
"""
from __future__ import annotations

import importlib.util
import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

from kdev.contract import Effects, ErrorCode, KdevError, Result
from kdev.registry import Arg, Context, Registry

P = "KDEV"
COMPONENTS = ("x_loads", "decode", "mma", "escape", "epilogue", "stream")
ALIASES = {"nox": "x_loads", "x": "x_loads", "nodec": "decode", "dec": "decode", "nomma": "mma", "noesc": "escape",
           "esc": "escape", "noep": "epilogue", "ep": "epilogue", "pure": "stream"}
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def norm_component(c: str) -> str:
    c = c.strip().lower()
    return ALIASES.get(c, c)


def _split_components(items: list[str] | None) -> list[str]:
    out: list[str] = []
    for it in items or []:
        for c in it.split(","):
            if c.strip():
                n = norm_component(c)
                if n not in out:
                    out.append(n)
    return out


def find_hook(base: str, explicit: str | None = None) -> str | None:
    if explicit:
        p = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isfile(p):
            raise KdevError(ErrorCode.NOT_FOUND, f"--hook not found: {explicit}", recovery="give a kdev_ablate.py")
        return p
    d = os.path.dirname(os.path.abspath(base))
    for _ in range(3):
        p = os.path.join(d, "kdev_ablate.py")
        if os.path.isfile(p):
            return p
        d = os.path.dirname(d)
    return None


def _load_hook(path: str) -> Any:
    spec = importlib.util.spec_from_file_location("kdev_ablate_hook", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, "ablate"):
        raise KdevError(ErrorCode.INVALID_REQUEST, f"{path}: no ablate(base, component, out_dir)",
                        recovery="def ablate(base, component, out_dir) -> path | {'path', 'knobs'} | None")
    return mod


def _parse_kv_list(items: list[str] | None, what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in [x for item in items or [] for x in item.split(",") if x.strip()]:
        k, eq, v = it.strip().partition("=")
        if not eq or not k or not v:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"bad {what} {it!r} (want NAME=VALUE)", recovery=f"{what} mma=f.py")
        out[norm_component(k)] = v
    return out


def plan(args: Namespace) -> dict[str, Any]:
    """Resolve the variant of every requested component (pure apart from running the hook)."""
    from kdev.timing.commands import _parse_value

    base = os.path.abspath(os.path.expanduser(args.file))
    if not os.path.isfile(base):
        raise KdevError(ErrorCode.NOT_FOUND, f"file not found: {args.file}", recovery="check the path")
    comps = _split_components(getattr(args, "component", None))
    explicit = _parse_kv_list(getattr(args, "variant", None), "--variant")
    knob_sets: dict[str, dict[str, Any]] = {}
    for s in getattr(args, "set", None) or []:
        c, sep, kv = s.partition(":")
        k, eq, v = kv.partition("=")
        if not sep or not eq or not k:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"bad --set {s!r} (want COMPONENT:K=V)",
                            recovery="e.g. --set mma:XS_NO_MMA=1")
        knob_sets.setdefault(norm_component(c), {})[k] = _parse_value(v)
    for c in list(explicit) + list(knob_sets):
        if c not in comps:
            comps.append(c)
    if getattr(args, "control", None):
        explicit["control"] = args.control
    if not comps:
        raise KdevError(ErrorCode.INVALID_REQUEST, "no component", recovery="--component " + ",".join(COMPONENTS))
    hook_path = find_hook(base, getattr(args, "hook", None))
    hook = None
    out_dir = os.path.join(os.path.dirname(base), "ablate_" + Path(base).stem)
    variants: list[dict[str, Any]] = []
    missing: list[str] = []
    want = comps + (["control"] if "control" in explicit or hook_path else [])
    for c in want:
        v: dict[str, Any] | None = None
        if c in explicit:
            v = {"path": os.path.abspath(os.path.expanduser(explicit[c])), "knobs": {}, "source": "--variant"}
        elif c in knob_sets:
            v = {"path": base, "knobs": knob_sets[c], "source": "--set"}
        elif hook_path:
            hook = hook or _load_hook(hook_path)
            os.makedirs(out_dir, exist_ok=True)
            got = hook.ablate(base, c, out_dir)
            if isinstance(got, str):
                v = {"path": os.path.abspath(got), "knobs": {}, "source": "hook"}
            elif isinstance(got, dict) and got.get("path"):
                v = {"path": os.path.abspath(got["path"]), "knobs": dict(got.get("knobs") or {}), "source": "hook"}
        if v is None:
            if c != "control":
                missing.append(c)
            continue
        if not os.path.isfile(v["path"]):
            raise KdevError(ErrorCode.NOT_FOUND, f"{c}: variant file not found: {v['path']}", recovery="check the path")
        v["component"] = c
        variants.append(v)
    if missing:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"no variant for component(s) {', '.join(missing)}",
                        details={"hook": hook_path},
                        recovery="--variant C=PATH, --set C:K=V, or a kdev_ablate.py hook next to the file "
                                 "(def ablate(base, component, out_dir) -> path)")
    if len(variants) + 1 > len(LABELS):
        raise KdevError(ErrorCode.INVALID_REQUEST, "too many variants", recovery="split the components")
    for i, v in enumerate(variants):
        v["label"] = LABELS[i + 1]
    return {"base": base, "hook": hook_path, "variants": variants,
            "labels": {v["label"]: v["component"] for v in variants}}


def _floors(items: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for it in items or []:
        for kv in it.split(","):
            k, eq, v = kv.partition("=")
            if eq:
                out[k.strip()] = float(v)
    return out


def _plan_lines(pl: dict[str, Any]) -> list[str]:
    lines = [f"  A  base      {pl['base']}"]
    for v in pl["variants"]:
        kn = " ".join(f"{k}={x}" for k, x in v["knobs"].items())
        lines.append(f"  {v['label']}  {v['component']:<9} {v['path']}" + (f"  [{kn}]" if kn else "") + f"  ({v['source']})")
    return lines


def _build(pl: dict[str, Any], ctx: Context) -> Result:
    from kdev.build.commands import cmd_build

    groups: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for v in [{"path": pl["base"], "knobs": {}}] + pl["variants"]:
        key = json.dumps(v["knobs"], sort_keys=True)
        g = groups.setdefault(key, (v["knobs"], []))
        if v["path"] not in g[1]:
            g[1].append(v["path"])
    last: Result | None = None
    for knobs, files in groups.values():
        ns = Namespace(files=files, variant=[f"{k}={x}" for k, x in knobs.items()] or None, defines="", jobs=8,
                       force=False, verbose=False, no_discover=False, clean_pending=False, discover_rounds=6)
        last = cmd_build(ns, ctx)
        if last.status != "ok":
            return last
    return last  # type: ignore[return-value]


def _run_dir_of(res: Result) -> str | None:
    d = (res.files or {}).get("dir")
    if d:
        return d
    if res.run_id:
        from kdev.build._runs import home

        p = home() / "runs" / res.run_id
        return str(p) if p.is_dir() else None
    return None


def finish_budget(res_doc: dict[str, Any], labels: dict[str, str], floors: dict[str, float], run_dir: str | None,
                  title: str = "") -> Result:
    from kdev.ablate import budget as B

    b = B.budget(res_doc.get("shapes") or [], "A", labels, floors)
    lines = [f"OK  ablate  {len(b['components'])} component(s), {len(b['shapes'])} shape(s)"
             + (f"  ab {res_doc.get('verdict', '')} run {res_doc.get('run_id')}" if res_doc.get("run_id") else "")]
    lines += B.render(b)
    files: dict[str, str] = {}
    if run_dir and os.path.isdir(run_dir):
        md = os.path.join(run_dir, "budget.md")
        Path(md).write_text(B.markdown(b, title), encoding="utf-8")
        Path(os.path.join(run_dir, "budget.json")).write_text(json.dumps(b, indent=1), encoding="utf-8")
        files = {"budget": md, "dir": run_dir}
    return Result(tool="ablate", status="ok", verdict="ok", run_id=res_doc.get("run_id"), summary={"budget": b},
                  files=files, lines=lines)


def cmd_ablate(args: Namespace, ctx: Context) -> Result:
    floors = _floors(getattr(args, "floor", None))
    if getattr(args, "from_run", None):
        from kdev.build._runs import home

        d = Path(args.from_run)
        if not d.is_dir():
            d = home(ctx) / "runs" / args.from_run
        rp = d / "result.json"
        if not rp.is_file():
            raise KdevError(ErrorCode.NOT_FOUND, f"no result.json in {d}", recovery="give an ab run id or dir")
        doc = json.loads(rp.read_text(encoding="utf-8"))
        labels = _parse_kv_list(getattr(args, "map", None), "--map") if getattr(args, "map", None) else {}
        labels = {k.upper(): v for k, v in labels.items()}
        if not labels and (d / "ablate.json").is_file():
            labels = json.loads((d / "ablate.json").read_text(encoding="utf-8")).get("labels", {})
        if not labels:
            raise KdevError(ErrorCode.INVALID_REQUEST, "which label is which component?",
                            recovery="--map B=x_loads,C=mma (A = base)")
        return finish_budget(doc, labels, floors, str(d), title=os.path.basename(str(doc.get("run_id") or d)))
    if not getattr(args, "file", None):
        raise KdevError(ErrorCode.INVALID_REQUEST, "FILE or --from-run", recovery="kdev ablate FILE --component mma")
    pl = plan(args)
    if getattr(args, "dry_run", False):
        return Result(tool="ablate", status="ok", verdict="plan", summary={"plan": pl},
                      lines=[f"PLAN  ablate  base + {len(pl['variants'])} variant(s) (hook: {pl['hook'] or '-'})"]
                      + _plan_lines(pl) + ["then: kdev build (all), one judge-mode ab A first, budget table"])
    if not getattr(args, "no_build", False):
        br = _build(pl, ctx)
        if br is not None and br.status != "ok":
            return br
    from kdev.timing.commands import _submit, build_ab_spec

    sets = [f"{v['label']}:{k}={x}" for v in pl["variants"] for k, x in v["knobs"].items()]
    ns = Namespace(**{k: v for k, v in vars(args).items() if k not in ("file", "set", "variant")})
    ns.files = [pl["base"]] + [v["path"] for v in pl["variants"]]
    ns.set = sets or None
    ns.judge_mode = not getattr(args, "fast", False)
    ns.run_name = ns.run_name if getattr(ns, "run_name", "") else "ablate_" + Path(pl["base"]).stem
    res = _submit(build_ab_spec(ns, ctx), ns, ctx)
    if res.error is not None or not res.shapes:
        return res
    rd = _run_dir_of(res)
    if rd and os.path.isdir(rd):
        Path(os.path.join(rd, "ablate.json")).write_text(json.dumps(pl, indent=1), encoding="utf-8")
    out = finish_budget(res.to_dict(), pl["labels"], floors, rd, title=os.path.basename(pl["base"]))
    out.fidelity = res.fidelity
    out.warnings = list(res.warnings or [])[:8]
    out.lines = _plan_lines(pl) + out.lines
    return out


def register(reg: Registry) -> None:
    from kdev.timing.commands import FIDELITY_ARGS, JUDGE_ARGS

    reg.command("ablate", "time ablation variants (one component removed each) vs the file in one judge-mode ab "
                "and print the excess budget (us each component costs per shape)", cmd_ablate,
                args=[Arg(("file",), "base file", type="path", nargs="?", default=None),
                      Arg(("--component",), "components: " + ",".join(COMPONENTS) + " (or any hook name; "
                          "comma-separated, repeatable)", action="append", default=None),
                      Arg(("--variant",), "COMPONENT=PATH ready ablation file (repeatable)", action="append",
                          default=None),
                      Arg(("--set",), "COMPONENT:K=V knob/define ablation of the base file (repeatable)",
                          action="append", default=None),
                      Arg(("--hook",), "per-problem kdev_ablate.py (default: next to FILE or 1-2 dirs up)",
                          type="path", default=None),
                      Arg(("--control",), "control variant (hooks compiled in, nothing removed): noise band",
                          type="path", default=None),
                      Arg(("--floor",), "SHAPE=US pure-read floor (repeatable/comma list): excess + remainder",
                          action="append", default=None),
                      Arg(("--rounds",), "ABBA blocks of processes", type="int", default=None),
                      Arg(("--fast",), "single-process ab (screening only)", action="store_true"),
                      Arg(("--no-build",), "skip kdev build (already built)", action="store_true"),
                      Arg(("--dry-run",), "print the variant plan only", action="store_true"),
                      Arg(("--from-run",), "re-render the budget of a finished ab run (id or dir)", default=None),
                      Arg(("--map",), "with --from-run: LABEL=COMPONENT (repeatable)", action="append",
                          default=None)]
                + FIDELITY_ARGS + JUDGE_ARGS,
                effects=Effects(gpu="queued", writes=("cache", "run_dir", "ledger")), gpu=True,
                long_running=True, package=P)
