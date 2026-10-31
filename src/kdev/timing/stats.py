"""Statistics: medians, paired ratios, bootstrap CIs, score CI, verdicts. Owner: P3. Pure Python (numpy
optional, used if importable). All randomness takes an explicit seed so results are reproducible."""
from __future__ import annotations

from typing import Callable, Sequence

from kdev.models import Interval


def median(xs: Sequence[float]) -> float:
    raise NotImplementedError("stats.median (P3)")


def paired_ratios(num: Sequence[float], den: Sequence[float], order: Sequence[int] | None = None
                  ) -> list[float]:
    """Per-rep ratios den/num style used by the judge: speedup = base/sub. With ``order`` (which side ran
    first per rep) pair order-swapped reps two at a time, as the judge does (TASK.md 'taken two repetitions
    at a time with the order swapped')."""
    raise NotImplementedError("stats.paired_ratios (P3)")


def bootstrap_ci(xs: Sequence[float], stat: Callable[[Sequence[float]], float] | None = None,
                 n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> Interval:
    """Percentile bootstrap CI of ``stat`` (default median)."""
    raise NotImplementedError("stats.bootstrap_ci (P3)")


def score_ci(per_shape_samples: dict[str, Sequence[float]], score_fn: Callable[[dict[str, float]], float],
             n_boot: int = 2000, seed: int = 0) -> Interval:
    """Propagate per-shape sample noise through the adapter's score formula by resampling reps."""
    raise NotImplementedError("stats.score_ci (P3)")


def verdict(ratio: Interval, noise_floor: float = 0.03, combined_se: float | None = None,
            suspect: bool = False) -> str:
    """faster | slower | same | suspect. Needs the CI to exclude 1 and |delta| >= max(noise_floor,
    2*combined_se). ``ratio`` is candidate speedup over the reference (>1 = faster)."""
    raise NotImplementedError("stats.verdict (P3)")


def max_ulp(a: Sequence[float], b: Sequence[float]) -> int:
    """Max ulp distance (used by check --repeat nondeterminism classification)."""
    raise NotImplementedError("stats.max_ulp (P3)")
