"""Judge integration, record-only (DESIGN 5.15, PLAN 8.1). Owner: P4.

kdev NEVER submits. ``okc_cli`` is the only module allowed to invoke the `okc` binary; `submit` is hard
denied there, and `check` is reachable only via ``pair.run_pair(allow_check=True)`` from the CLI flag
``kdev judge pair --allow-check`` (not exposed over MCP). tests/unit/test_p4_safety.py greps the tree.
"""
