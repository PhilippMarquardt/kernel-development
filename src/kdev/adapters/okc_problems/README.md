One `<problem>.toml` per OKC problem (P7 writes them from each TASK.md "Shapes" table and the local
harnesses' TESTS/EXTRA lists: ah.py, cq_run.py, hp.py, hm.py, ph.py in arcagi3/okc/muse_*/muse_*/).

```toml
schema = "kdev.okc_problem/1"
problem = "muse_attention_decode"
kernel_fn_args = 1                 # 1: kernel_fn(data); 2: kernel_fn(data, state) with prepare(weights)
input_variance = { routing = false, lengths = true }
[protocol]
overhead_us = 4.0
judge_reps = 6
budget_s = 900
[[shapes]]                         # scored
id = "d1/sliding"; load = "d1"; weight = 46
case = { batch = 1, ctx = 80000, heads = "tp8", layer = "sliding", page = 1, q_len = 1, dist = "soft" }
[[tests]]
id = "t0"; case = { batch = 1, ctx = 10, heads = "tp8", layer = "sliding" }
[[extra]]
id = "x0"; case = { batch = 8, ctx = 80000, heads = "tp8", layer = "full", q_len = 8, page = 1 }
[comply]
precision = { router = "fp32" }
forbid_patterns = ['\.item\(\)', 'torch\.cuda\.synchronize']
```
