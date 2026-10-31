One `<problem>.toml` per OKC problem (P7). Shapes/weights/loads match each TASK.md "Shapes" table exactly (unit
test + manual check against arcagi3/okc/muse_*/muse_*/TASK.md), task seeds/run counts come from the OKC API
benchmark list and judge records, tests from TASK.md, extra (held-out, nearby) shapes from the local harnesses
(ah.py, cq_run.py, hp.py, hm.py, ph.py) plus nearby sizes. Generator: none kept; edit by hand.

```toml
schema = "kdev.okc_problem/1"
problem = "muse_attention_decode"
gpu = "h200"
kernel_fn_args = 1          # 1: kernel_fn(data); 2: kernel_fn(data, state) with state = prepare(weights)
check_style = "collect"     # collect: check_implementation(data, collect(data, out)[, w]); direct: (data, out[, w])
weights = false             # reference.generate_weights(WEIGHTS_SEED) passed to prepare()
case_fields = false         # reference.case_fields(case, weights) merged into every input (decode/prefill)
default_seed = 1234
input_variance = { routing = false, lengths = true }   # any true -> 5 seeds by default
search_dirs = ["/root/okc/muse_attention_decode/muse_attention_decode"]
# router_seed = 123         # optional: generate_weights(..., router_seed=) (judge picks one per run)

[protocol]                  # TimingProtocol fields
overhead_us = 4.0
judge_reps = 60
budget_s = 900
prepare_budget_s = 300

[[shapes]]                  # scored
id = "d1/sliding"
load = "decode 1"           # judge load name (score: geomean over loads of weighted sums)
weight = 46
seed = 601                  # task seed (API benchmark list); judge JSONs carry the per-run seeds
repeats = 60                # judge timed runs for this shape
floor_us = 0.1
case = { batch = 1, ctx = 80000, heads = "tp8", layer = "sliding", page = 1, dist = "soft" }

[[tests]]
id = "t0"
case = { batch = 1, ctx = 10, heads = "tp8", layer = "sliding" }

[[extra]]                   # held-out nearby shapes (`kdev check --extra-shapes`)
id = "x0"
case = { batch = 8, ctx = 80000, heads = "tp8", layer = "full", q_len = 8, page = 1 }

[comply]
precision = { kv_cache = "fp8_e4m3" }
forbid_patterns = ['\.item\(\)', 'torch\.cuda\.synchronize']
[comply.every_key]          # dynamic probe: planted value bands must be attended
enabled = true
[comply.routing]            # decode/prefill: router precision probe at T <= fp32_max_tokens
fp32_max_tokens = 8
probe_shapes = ["d1/sliding"]
```
