# kdev MCP server (P8)

`pip install -e ".[mcp]"`, then `kdev mcp [--profile compact|full]` (or `python -m kdev.mcp`), stdio transport.
Claude Code: `claude mcp add kdev -- kdev mcp` (on the pod; or `ssh pod kdev mcp` as the command).
`kdev mcp tools [--profile P]` prints the generated tool table and its context cost without the SDK.

## Contract
- Tools are generated from the command registry (`kdev.mcp.server.build_tools`). A call's arguments become an
  argv for that command's CLI parser, so types/choices/defaults are the CLI's; the handler runs with
  `Context(via="mcp")`. The result is the `kdev.result/1` envelope, identical to `kdev <cmd> --json`, in both
  `structuredContent` and the text content. `isError` = `status == "error"`.
- Over the inline budget (24 KB) `shapes`/`dispatch`/`summary` are dropped with a warning naming
  `kdev_show run_id=... section=result`; page with `kdev_show` or resource `kdev://runs/{run_id}/{section}?offset=&limit=`.
- Profiles: `compact` (default, 12 tools: doctor, build, check, time, ab, sass{analyze,diff,regs}, timeline,
  ledger{query,tried,best,judged,import,export}, idea{list,show,add,claim,release,evidence,resolve},
  job{status,wait,list,logs,cancel,queue,offenders}, show, run{command, argv}); `full` = one tool per command.
- GPU tools always enqueue detached and return `job_id`/`run_id`; `wait_s` (<= 120) polls `job wait` in 10 s
  chunks with progress notifications.
- Errors: stable `error.code` (PLAN 7); an empty `recovery` is filled from `kdev.mcp.server.RECOVERY`.

## Never exposed (`kdev.mcp.schema.HIDDEN_*`, stricter than the registry flags)
`judge pair` (and `--allow-check`), any command whose effects have `quota != never` or `network == okc_check`,
`lock run`, `serve*`, `remote*`, `sync*`, `job submit`, `dashboard watch/serve`, `mcp`; args `ledger query --sql`,
`--home`, `--json`, `--quiet`, `--direct`, `--detach`, REMAINDER argv. `kdev_run` applies the same rules.
Nothing reachable over MCP can submit to or check on the judge.

## Implementation note
Uses the SDK's low-level `mcp.server.lowlevel.Server` (not FastMCP), because input schemas are generated from
the registry rather than from Python signatures. `kdev.mcp` modules import the SDK only inside
`build_server`/`main`.
