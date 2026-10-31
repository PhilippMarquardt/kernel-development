"""Shared data model (FROZEN after Phase 0; see docs/PLAN.md section 3).

Dataclasses that cross package boundaries: job specs, per-shape results, toolchain/GPU descriptions.
Each has ``to_dict``/``from_dict``. Change only via docs/CHANGES.md + coordinator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, TypeVar

JOB_SCHEMA = "kdev.job/1"

JOB_KINDS = ("build", "check", "time", "ab", "sweep", "calib", "sanitize", "comply", "timeline", "profile",
             "dispatch", "floor_calibrate", "exec", "discover")
BUILD_LANE_KINDS = ("build",)

T = TypeVar("T")


class _Dict:
    """Mixin: dict round-trip ignoring unknown keys (forward compatible)."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls: type[T], d: dict[str, Any]) -> T:
        names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class FileRef(_Dict):
    """One input file of a job (a submission, the baseline, or a sweep variant)."""

    path: str
    label: str = "A"
    md5: str = ""
    knobs: dict[str, Any] = field(default_factory=dict)   # @override knob dict / --variant K=V
    defines: list[str] = field(default_factory=list)       # extra -D flags
    judge_json: str | None = None                          # for calib: path to `okc status --json`
    role: str = "sub"                                      # sub | base | variant


@dataclass
class TimingParams(_Dict):
    """Timing/check parameters carried in ``JobSpec.params``. Defaults are the judge protocol."""

    reps: int = 40
    seeds: int | None = None            # None = adapter default (5 if input_variance has routing/lengths)
    seed_list: list[int] | None = None  # explicit seeds (overrides `seeds`)
    shapes: str | None = None           # substring filter over shape_id / case
    order: str = "abba"                 # abba | rotate
    flush: str = "read"                 # read | write | none   (non-read = opt-out)
    flush_bytes: int | None = None      # None = GPU profile flush_bytes (256 MB on H200)
    graph: bool = True                  # False = --eager (opt-out)
    overhead: str = "judge"             # judge (4.0 us const) | local | none
    base: str | None = "baseline"       # baseline | none | path
    repeat: int = 1                     # check --repeat
    mode: str = "both"                  # check: graph | eager | both
    allow_jit: bool = False             # opt-out: compiling inside the GPU lane
    extra: dict[str, Any] = field(default_factory=dict)  # tool-specific (grid, tool=memcheck, marks, ...)


@dataclass
class JobSpec(_Dict):
    """A unit of work for the service (``kdev.job/1``)."""

    kind: str
    files: list[FileRef] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    problem: str | None = None
    adapter: str | None = None
    job_id: str = ""
    lane: str = ""                      # build | gpu ("" = derived from kind)
    agent: str = ""
    priority: int = 0
    timeout_s: int = 1800
    max_hold_s: int = 1800
    run_name: str = ""
    submitted_utc: str = ""
    requires_prebuilt: bool = True
    parent_job: str | None = None
    schema: str = JOB_SCHEMA

    def resolved_lane(self) -> str:
        return self.lane or ("build" if self.kind in BUILD_LANE_KINDS else "gpu")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JobSpec":
        d = dict(d)
        files = [FileRef.from_dict(f) if isinstance(f, dict) else f for f in d.pop("files", [])]
        names = {f.name for f in fields(cls)}
        spec = cls(**{k: v for k, v in d.items() if k in names and k != "files"})
        spec.files = files
        return spec


@dataclass
class JobStatus(_Dict):
    """Row view of a job in the service queue."""

    job_id: str
    state: str                  # queued | running | done | failed | cancelled | timeout
    lane: str
    kind: str
    agent: str = ""
    position: int | None = None  # queue position when queued
    run_id: str | None = None
    run_dir: str | None = None
    submitted_utc: str = ""
    started_utc: str | None = None
    finished_utc: str | None = None
    exit_code: int | None = None
    error_code: str | None = None
    waited_s: float | None = None


@dataclass
class Sample(_Dict):
    """One timed replay (a line of raw.jsonl)."""

    shape_id: str
    label: str
    seed: int
    rep: int
    order_pos: int
    us_raw: float
    us: float


@dataclass
class Interval(_Dict):
    median: float
    ci_lo: float
    ci_hi: float
    n: int = 0


@dataclass
class FileShapeTiming(_Dict):
    label: str
    md5: str
    median_us: float
    ci_lo_us: float
    ci_hi_us: float
    n: int
    seeds: list[int] = field(default_factory=list)
    per_seed_median_us: list[float] = field(default_factory=list)


@dataclass
class ShapeResult(_Dict):
    """Per-shape entry of ``Result.shapes``."""

    shape_id: str
    weight: float = 1.0
    load: str = ""
    per_file: list[dict[str, Any]] = field(default_factory=list)   # FileShapeTiming dicts
    ratio: dict[str, Any] | None = None    # {num, den, median, ci_lo, ci_hi}
    verdict: str = ""                      # faster | slower | same | incorrect | suspect
    check_status: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class Toolchain(_Dict):
    """Detected or pinned toolchain (see toolchain.lock, PLAN 3.9)."""

    nvcc_version: str = ""
    cuda_home: str = ""
    nvcc: str = ""
    ptxas: str = ""
    cuobjdump: str = ""
    nvdisasm: str = ""
    torch: str = ""
    triton: str = ""
    python: str = ""
    env: dict[str, str] = field(default_factory=dict)
    extra: dict[str, str] = field(default_factory=dict)
    source: str = ""


@dataclass
class KernelInfo(_Dict):
    """ptxas -v / SASS facts about one kernel."""

    name: str
    regs: int | None = None
    spill_store: int = 0
    spill_load: int = 0
    stack: int = 0
    smem_static: int = 0
    barriers: int | None = None
    cmem: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)     # verbatim C751x etc.
    sass_instr: int | None = None


@dataclass
class Check(_Dict):
    """One doctor line."""

    name: str
    status: str            # PASS | WARN | FAIL | SKIP
    detail: str = ""
    hint: str = ""


@dataclass
class GpuState(_Dict):
    """nvidia-smi snapshot."""

    sm_mhz: int | None = None
    mem_mhz: int | None = None
    power_w: float | None = None
    temp_c: int | None = None
    util_pct: int | None = None
    name: str = ""
    uuid: str = ""
    ts_utc: str = ""
