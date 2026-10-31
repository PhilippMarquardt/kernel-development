"""Judge-faithful timing engine (protocol kdev-tp/1), statistics and calibration. Owner: P3.

Pure modules (no torch, unit-testable): stats, schedule, seeds, overhead (math part), calib (parsing/compare).
GPU modules (lazy ``import torch`` inside functions only): flush, graph, engine (TorchTimer).
"""
