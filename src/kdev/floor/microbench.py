"""Microbenchmark suite for `kdev floor calibrate` (GPU, worker kind='floor_calibrate'). Owner: P6.

Ports the ideas of okc fb.py/fl_src.py/tlb.cu/ic.py: null kernel, empty/1-kernel graph, dependent DRAM
chase (1 CTA, all SMs), L2 hit, kernel boundary (with/without PDL), extra graph node, cooperative launch,
MEMBAR.GPU, cold atom.acq_rel, barrier.cluster, 200 MB stream. All after a read flush, graph replay, median.
"""
from __future__ import annotations

from typing import Any

from kdev.contract import Result
from kdev.models import JobSpec

BENCHES = ("null_kernel", "empty_graph", "one_kernel_graph", "dram_chase_1cta", "dram_chase_all",
           "l2_hit", "kernel_boundary", "kernel_boundary_pdl", "graph_node", "coop_launch", "membar_gpu",
           "atom_acqrel", "barrier_cluster", "stream_200mb")


def run_calibrate(spec: JobSpec, run_dir: Any) -> Result:
    raise NotImplementedError("microbench.run_calibrate (P6)")
