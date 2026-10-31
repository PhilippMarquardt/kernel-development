"""Statistics: medians, paired ratios, bootstrap CIs, score CI, verdicts. Owner: P3. Pure Python (stdlib
only; fast enough for 10k resamples of <= a few hundred reps). All randomness takes an explicit seed so
results are reproducible.

Judge facts these functions reproduce (verified on 5 judge JSONs, tests/unit/test_p3_stats.py):
- per-rep ratio r_i = (base_raw_i - ovh) / (sub_raw_i - ovh), ovh = 2*one_kernel - two_kernel graph replay;
- shape speedup = median over k of sqrt(r_2k * r_2k+1) (reps taken two at a time, order swapped);
- shape sub time = base median (overhead subtracted) / speedup;
- problem speedup = geomean over loads of sum(w*base)/sum(w*sub); its CI is widened to at least +-1.5 %.
"""
from __future__ import annotations

import math
import random
import struct
from typing import Callable, Sequence

from kdev.models import Interval

JUDGE_MIN_CI_HALF = 0.015   # the judge reports at least +-1.5 % around the problem speedup


def median(xs: Sequence[float]) -> float:
    """Median (mean of the two middle values for even n). NaN for an empty sequence."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    h = n // 2
    return s[h] if n % 2 else 0.5 * (s[h - 1] + s[h])


def quantile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile (numpy 'linear'), q in [0, 1]."""
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = (len(s) - 1) * min(1.0, max(0.0, q))
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def pair_geomean(ratios: Sequence[float]) -> list[float]:
    """Judge pairing: sqrt(r[2k] * r[2k+1]) for consecutive reps (a trailing odd rep is dropped)."""
    return [math.sqrt(ratios[i] * ratios[i + 1]) for i in range(0, len(ratios) - 1, 2)
            if ratios[i] > 0 and ratios[i + 1] > 0]


def paired_ratios(num: Sequence[float], den: Sequence[float], order: Sequence[int] | None = None
                  ) -> list[float]:
    """Per-rep ratios den/num style used by the judge: speedup = base/sub. With ``order`` (which side ran
    first per rep) pair order-swapped reps two at a time, as the judge does (TASK.md 'taken two repetitions
    at a time with the order swapped'): result k = sqrt(r[a] * r[b]) for consecutive reps a, b whose first
    side differs. Without ``order`` the per-rep ratios den[i]/num[i] are returned unpaired.

    ``num`` = candidate times (sub), ``den`` = reference times (base), overhead already subtracted."""
    if len(num) != len(den):
        raise ValueError(f"paired_ratios: length mismatch {len(num)} vs {len(den)}")
    r = [d / n if n > 0 else float("inf") for n, d in zip(num, den)]
    if order is None:
        return r
    if len(order) != len(r):
        raise ValueError("paired_ratios: order length mismatch")
    out: list[float] = []
    i = 0
    while i + 1 < len(r):
        if order[i] != order[i + 1]:
            a, b = r[i], r[i + 1]
            if a > 0 and b > 0 and math.isfinite(a) and math.isfinite(b):
                out.append(math.sqrt(a * b))
            i += 2
        else:            # not order-swapped (e.g. rotate with 3+ files): skip one rep and re-sync
            i += 1
    return out


def bootstrap_ci(xs: Sequence[float], stat: Callable[[Sequence[float]], float] | None = None,
                 n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> Interval:
    """Percentile bootstrap CI of ``stat`` (default median). ``Interval.median`` is stat(xs)."""
    f = stat or median
    xs = list(xs)
    n = len(xs)
    if n == 0:
        nan = float("nan")
        return Interval(nan, nan, nan, 0)
    point = f(xs)
    if n == 1:
        return Interval(point, point, point, 1)
    rng = random.Random(seed)
    boots = [f(rng.choices(xs, k=n)) for _ in range(n_boot)]
    return Interval(point, quantile(boots, alpha / 2), quantile(boots, 1 - alpha / 2), n)


def score_ci(per_shape_samples: dict[str, Sequence[float]], score_fn: Callable[[dict[str, float]], float],
             n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
             stat: Callable[[Sequence[float]], float] | None = None,
             min_half_width: float = 0.0) -> Interval:
    """Propagate per-shape sample noise through the adapter's score formula by resampling reps.

    ``per_shape_samples[shape]`` are the per-pair speedups (or any samples); each bootstrap draw resamples
    every shape independently, takes ``stat`` (default median) per shape and calls ``score_fn`` on the dict.
    ``min_half_width`` widens the CI to at least +-that fraction of the point value (judge: 0.015)."""
    f = stat or median
    shapes = [s for s, v in per_shape_samples.items() if len(v) > 0]
    if not shapes:
        nan = float("nan")
        return Interval(nan, nan, nan, 0)
    data = {s: list(per_shape_samples[s]) for s in shapes}
    point = score_fn({s: f(v) for s, v in data.items()})
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(score_fn({s: f(rng.choices(v, k=len(v))) for s, v in data.items()}))
    lo, hi = quantile(boots, alpha / 2), quantile(boots, 1 - alpha / 2)
    if min_half_width > 0:
        lo = min(lo, point * (1 - min_half_width))
        hi = max(hi, point * (1 + min_half_width))
    return Interval(point, lo, hi, min(len(v) for v in data.values()))


def ratio_se(ci: Interval, alpha: float = 0.05) -> float:
    """Approximate standard error from a (1-alpha) CI half width (normal approximation)."""
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _z(1 - alpha / 2)
    return abs(ci.ci_hi - ci.ci_lo) / (2 * z)


def _z(p: float) -> float:
    # Acklam-free simple inverse via bisection on erf (only used for non-default alpha)
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def verdict(ratio: Interval, noise_floor: float = 0.03, combined_se: float | None = None,
            suspect: bool = False) -> str:
    """faster | slower | same | suspect. Needs the CI to exclude 1 and |delta| >= max(noise_floor,
    2*combined_se). ``ratio`` is candidate speedup over the reference (>1 = faster)."""
    if suspect:
        return "suspect"
    m = ratio.median
    if m is None or not math.isfinite(m):
        return "same"
    thr = max(noise_floor, 2 * combined_se) if combined_se is not None else noise_floor
    delta = m - 1.0
    if ratio.ci_lo > 1.0 and delta >= thr:
        return "faster"
    if ratio.ci_hi < 1.0 and -delta >= thr:
        return "slower"
    return "same"


def _f32_ordered(x: float) -> int:
    """float32 bit pattern mapped onto a monotonically ordered integer line."""
    i = struct.unpack("<i", struct.pack("<f", x))[0]
    return i if i >= 0 else -(i & 0x7FFFFFFF)


def max_ulp(a: Sequence[float], b: Sequence[float]) -> int:
    """Max ulp distance (float32 ulps; used by check --repeat nondeterminism classification).
    NaN vs non-NaN counts as 2**31; NaN vs NaN as 0."""
    if len(a) != len(b):
        raise ValueError("max_ulp: length mismatch")
    worst = 0
    for x, y in zip(a, b):
        xn, yn = math.isnan(x), math.isnan(y)
        if xn or yn:
            if xn != yn:
                worst = max(worst, 2**31)
            continue
        worst = max(worst, abs(_f32_ordered(x) - _f32_ordered(y)))
    return worst


def spread(xs: Sequence[float]) -> float:
    """(max - min) / median of a list (per-seed spread report); 0 for < 2 values."""
    if len(xs) < 2:
        return 0.0
    m = median(xs)
    return (max(xs) - min(xs)) / m if m else float("nan")
