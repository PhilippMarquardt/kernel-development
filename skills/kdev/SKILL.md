---
name: kdev
description: Workflow for writing and optimising GPU kernels with the kdev toolkit (judge-faithful timing, SASS audits, ledger, idea board, GPU queue). Use whenever you build, time, compare, audit or claim anything about a kernel on the GPU pod - including before saying a kernel is "at the limit".
---

# kdev: kernel optimisation workflow

kdev is the only way you build, time and compare kernels. It encodes what our OKC agents kept getting wrong
(okc/LESSONS.md): local numbers that lied (wrong toolchain, write-flush, contended GPU, eager timing), "at
the limit" claims nobody checked, and knowledge lost when an agent was retired. Follow the order below; skip
a step only if the ledger shows it was already done on the *same* file (md5) under the current protocol.

Every command prints a <= 40-line summary (verdict first, then `run: <id>`), or one JSON object with
`--json`. The run id is your evidence id: cite it in notes and ideas. Read big outputs with
`kdev show RUN --section S --offset N --limit M` - never `cat` a run's files. Errors carry a code and a
one-line `recovery`; do what it says (e.g. `needs_build` -> `kdev build FILE` first).

## 0. Before you touch anything (ledger first)

1. `kdev doctor` - GPU, toolchain pin (judge nvcc), lock, backends. Fix every FAIL before timing.
2. `kdev ledger tried FILE` (or `kdev ledger tried "idea text" --knobs K=V`) - has this exact file, a
   comment-only variant, or this knob set been tried? Read the hits; do not repeat a measured experiment.
3. `kdev ledger best --problem P` and `kdev judge ledger` - current judged best and its same-hour baseline.
4. `kdev idea list --status open` - claim what you work on with `kdev idea claim ID` (one owner per idea);
   add new hypotheses with `kdev idea add "TITLE" --hypothesis H --gain PCT`.
5. `kdev dispatch FILE` - confirm which kernels actually run per scored shape. Files carry overwritten entry
   points and embedded alternatives; it is easy to optimise a path that never runs.

## 0b. Talk to the other agents (board + docs)

Set `KDEV_AGENT=<your id>`. Run the start checks below at the start of your work and again before every GPU job.

- **Start checks.** `kdev board inbox` shows posts to you, @mentions, replies and claims that overlap yours.
  Clear it with `kdev board inbox --ack`. Then run `kdev board read --unread --problem P` and
  `kdev board claims --problem P`.
- **Before you start a piece of work,** post a claim:
  `kdev board post claim "plan" --problem P --files F1,F2 [--shapes S]`.
  If it prints `CONFLICT`, do not duplicate the work. Ask the owner first:
  `kdev board post ask "@owner ..." --reply-to ID`.
- **After every result,** post a finding: `kdev board post finding "variant: x0.98 [0.97,0.99]" --problem P --ref RUN_ID`.
  Add `--doc notes --section "Round N <id>"` to also file it in the problem notes.
- **When you finish,** post `kdev board post done "summary" --problem P --ref RUN_ID`. This closes your claim.
- **Shared knowledge** goes into `kdev doc append notes --problem P --section S "evidence" --ref RUN_ID`. Never edit
  NOTES.md, LESSONS.md or TRACKER.md by hand: hand edits overwrote each other's sections. Read the notes with
  `kdev doc toc|show|search`. Get the rules with `kdev doc checklist`.

## 1. Judge-anchored baseline

- `kdev time FILE` times one file against the problem baseline with the judge protocol: CUDA-graph replay,
  read flush (256 MB), 4.0 us overhead, adapter seeds (>= 5 when routing/lengths vary), exclusive GPU.
- `kdev calib FILE:JUDGE.json` compares local per-shape speedups with a recorded judge run. Any `DISAGREE`
  blocks decisions based on local A/B for that problem until it is explained.
- The judge is the truth. Judge checks are run by the coordinator only (agents never submit or check); ask
  for a check of your candidate **and** of the unmodified base in the same hour, then
  `kdev judge record JOB.json` so the result lands in the ledger.

## 2. Cheap knobs, exhaustively, before any rewrite

- `kdev sweep FILE --grid "TILE=64,128 STAGES=2,3,4"` prebuilds every variant in parallel (no compile under
  the GPU lock), quick-checks, then runs batched interleaved A/B against the base. Search the whole cheap
  space: library algorithms, tile/stage/split plans, launch configs, existing `@override` knobs.
- Compare two concrete files with `kdev ab A.py B.py` (ABBA/rotate interleave, paired-ratio CIs). Verdicts
  `faster|slower|same` come from the CI, not from a single median. Use `--set B:K=V` for knob variants.

## 3. SASS audit of the top-time kernel

- `kdev build FILE` (pinned toolchain; ptxas regs/spills/C751x) then
  `kdev sass FILE --kernel RE --loops --audit pdl,spills,serialize,repeat` on the kernel that dominates time.
  Account for every instruction in the hot loop per element: repeated work, redundant decode, spills
  (LDL/STL), serialisation warnings (C7510-C7515), missing expected opcodes (`--audit expect=HGMMA`).
- PDL kernels: the `pdl` audit must show no producer-data load (LDG...) hoisted above ACQBULK
  (griddepcontrol.wait); a hoisted load gives intermittent wrong results.
- `kdev sass diff A B` after every change you expect to alter the inner loop; `kdev regs FILE` for occupancy.
- `kdev floor --problem P --shape S` gives the roofline / latency-skeleton floor; `kdev timeline FILE` shows
  where per-CTA time goes when a kernel is latency-bound.

## 4. Only then: rewrites

Structural changes (fusion, new kernels, megakernels) come after steps 1-3 and need an expected gain of
>= 5 % (ignore < 3 % items unless nearly free). Each rewrite goes through the same loop: `kdev check`,
`kdev ab` against the current best, `kdev sass` of the new hot loop.

## Rules that are never optional

- **Always measure with `kdev time` / `kdev ab`.** No hand-written timing loops, no bare `torch.cuda.Event`
  scripts, no timing outside the GPU queue. Opt-outs (`--flush write`, `--eager`, `--allow-jit`) are printed
  in CAPS, recorded, and make the run `not comparable` - never promote on them.
- **Correctness before speed:** `kdev check FILE` (fresh seeds, extra shapes, eager + graph, repeats) on
  every candidate, and `kdev comply FILE` (precision per op and size, no cross-call caching, no work outside
  the timed call) before anything becomes a best. `kdev sanitize FILE` when results are intermittent.
- **Never claim "at the limit" / "at the floor" without evidence.** A limit claim needs a `kdev sass` run
  of the hot loop AND a `kdev sweep` run of the cheap knobs, both attached:
  `kdev note "CLAIM" --kind limit --evidence RUN_SASS,RUN_SWEEP`. The board rejects confirming a `limit`
  idea without both, and the dashboard lists unverified claims. Treat inherited limit claims as unverified
  until you have re-checked them.
- **Evidence over conclusions:** every result you report names its run id(s). A dead idea records the exact
  variant tested (`kdev idea resolve ID --status refuted --resolution "TILE=256 STAGES=4: x0.97 [0.96,0.98],
  run R"`), so a different variant is not wrongly skipped. Attach runs with
  `kdev idea evidence ID --run RUN --supports for|against`.
- **GPU etiquette:** GPU commands enqueue; use `--detach` + `kdev job wait JOB` for long runs and keep
  working meanwhile. Never start CUDA work outside kdev; `kdev queue status` shows holders and offenders.
- **Outside world:** check `kdev dashboard build --scrape` (or `kdev okc leaderboard`) for the per-problem gap
  to board #1/#10 and per-shape time shares; re-scan public code and notes every 3 h and forward techniques.

## Brief checklist (paste into every optimisation agent prompt)

1. `kdev doctor`; `kdev ledger tried FILE`; `kdev idea list --status open`; claim your idea.
2. Judge-anchored baseline (`kdev time`, `kdev calib`); same-hour base judged next to any candidate.
3. Exhaustive cheap knobs (`kdev sweep`) before rewrites; compare with `kdev ab` only.
4. SASS audit of the top-time kernel (`kdev sass --loops --audit pdl,spills,serialize,repeat`).
5. No "at the limit" without sass + sweep run ids attached (`kdev note --kind limit`).
6. `kdev check` + `kdev comply` before promoting anything.
7. Write evidence (run ids, exact variant, numbers) to the idea board, not just conclusions.
8. `kdev board inbox` + `kdev board read --unread` at start and before each GPU job; `kdev board post claim ...
   --files F` before starting; `finding`/`done` with `--ref RUN` after; notes only via `kdev doc append`.

## MCP clients

`kdev mcp` exposes the same commands as MCP tools (compact profile: `kdev_doctor`, `kdev_build`, `kdev_check`,
`kdev_time`, `kdev_ab`, `kdev_sass`, `kdev_timeline`, `kdev_ledger`, `kdev_idea`, `kdev_job`, `kdev_board`,
`kdev_doc`, `kdev_show`, `kdev_run`) with identical JSON results. GPU tools return a job id unless `wait_s` (<= 120) is given.
Judge checks, raw lock commands, sync/remote and raw SQL are not available over MCP.
