"""Device-only compile through NVRTC (ctypes; no CUDA context, no host compiler). P5, private.

Used for standalone ``.cu`` builds when nvcc cannot run (Windows without cl.exe) or when asked explicitly
(``backend='nvrtc'``). Port of okc muse_moe_w4a16 tools/offline/nvrtc_dev.py.
"""
from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path


class NvrtcUnavailable(RuntimeError):
    pass


def _lib_candidates(cuda_home: str) -> list[str]:
    h = Path(cuda_home)
    pats = [h / "bin" / "x86_64" / "nvrtc64_*.dll", h / "bin" / "nvrtc64_*.dll", h / "lib64" / "libnvrtc.so*",
            h / "lib" / "libnvrtc.so*", h / "lib" / "x64" / "nvrtc64_*.dll"]
    out: list[str] = []
    for p in pats:
        out += sorted(x for x in glob.glob(str(p)) if "builtins" not in os.path.basename(x)
                      and not x.endswith(".alt.so"))
    return out


def _load(cuda_home: str):
    libs = _lib_candidates(cuda_home)
    if not libs:
        raise NvrtcUnavailable(f"no NVRTC library under {cuda_home}")
    lib = libs[-1]
    d = os.path.dirname(lib)
    if os.name == "nt":
        os.add_dll_directory(d)  # type: ignore[attr-defined]
        for b in glob.glob(os.path.join(d, "nvrtc-builtins64_*.dll")):
            ctypes.CDLL(b)
    return ctypes.CDLL(lib)


def compile_cubin(src: str, out_dir: str, *, cuda_home: str, arch: str = "sm_90a", flags: list[str] | None = None,
                  name: str = "k.cu", std: str = "c++17") -> tuple[int, str, str]:
    """(rc, log, cubin_path). rc 0 = ok. The log contains ptxas -v output (``--ptxas-options=-v``)."""
    nv = _load(cuda_home)
    os.makedirs(out_dir, exist_ok=True)
    prog = ctypes.c_void_p()
    r = nv.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), name.encode(), 0, None, None)
    if r != 0:
        return r, f"nvrtcCreateProgram failed: {r}", ""
    inc = str(Path(cuda_home) / "include")
    opts = [f"-arch={arch}", f"-std={std}", "--ptxas-options=-v", "-lineinfo", "-default-device", f"-I{inc}",
            "-D__CUDACC_RTC_OK__=1", *(flags or [])]
    arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
    rc = nv.nvrtcCompileProgram(prog, len(opts), arr)
    n = ctypes.c_size_t()
    nv.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value)
    nv.nvrtcGetProgramLog(prog, buf)
    log = buf.value.decode(errors="replace")
    cubin = ""
    if rc == 0:
        nv.nvrtcGetCUBINSize(prog, ctypes.byref(n))
        cub = ctypes.create_string_buffer(n.value)
        nv.nvrtcGetCUBIN(prog, cub)
        cubin = os.path.join(out_dir, Path(name).stem + ".cubin")
        with open(cubin, "wb") as f:
            f.write(cub.raw)
    nv.nvrtcDestroyProgram(ctypes.byref(prog))
    return rc, log, cubin
