---
name: kdev
description: Workflow for writing and optimising GPU kernels with the kdev toolkit (judge-faithful timing, SASS audits, ledger, idea board). Use when working on a kernel optimisation problem on a GPU pod.
---

# kdev workflow (Phase-0 placeholder; P8 writes the real skill)

Order (LESSONS 2): judge-anchored baseline -> exhaustive knob search (`kdev sweep`) -> SASS audit of the
top-time kernel (`kdev sass --loops --audit pdl,spills,serialize,repeat`) -> only then rewrites.

Before anything: `kdev doctor`, `kdev ledger tried FILE`, `kdev idea list --status open`.
