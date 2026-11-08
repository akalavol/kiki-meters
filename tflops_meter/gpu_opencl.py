"""GPU via OpenCL : crête de calcul FMA (tous fabricants) et débit VRAM hors NVIDIA."""
from __future__ import annotations

import ctypes
import time

from . import opencl

_NV_CORES_PER_SM = {
    (3, 0): 192, (3, 5): 192, (3, 7): 192, (5, 0): 128, (5, 2): 128, (5, 3): 128,
    (6, 0): 64, (6, 1): 128, (6, 2): 128, (7, 0): 64, (7, 2): 64, (7, 5): 64,
    (8, 0): 64, (8, 6): 128, (8, 7): 128, (8, 9): 128, (9, 0): 128,
    (10, 0): 128, (10, 1): 128, (11, 0): 128, (12, 0): 128, (12, 1): 128,
}


def gpu_devices():
    try:
        return [d for d in opencl.list_devices() if d.is_gpu]
    except opencl.OpenCLError:
        return []


def theoretical_fp32(dev) -> float | None:
    """Crête FP32 vectorielle (TFLOPS) = unités × voies × 2 (FMA) × fréquence boost."""
    if not dev.clock_mhz or not dev.compute_units:
        return None
    vendor = dev.vendor.lower()
    lanes = None
    if dev.cc:
        lanes = _NV_CORES_PER_SM.get(dev.cc) or (128 if dev.cc[0] >= 8 else None)
    elif "advanced micro devices" in vendor or "amd" in vendor:
        lanes = 64
    elif "intel" in vendor:
        lanes = 8
    if lanes is None:
        return None
    return dev.compute_units * lanes * 2 * dev.clock_mhz * 1e6 / 1e12


_TYPES = {
    # précision : (type OpenCL, voies, pragma, cast depuis float)
    "FP32": ("float", 1, "", "({t})({v})"),
    "FP64": ("double", 1, "#pragma OPENCL EXTENSION cl_khr_fp64 : enable", "({t})({v})"),
    "FP16": ("half2", 2, "#pragma OPENCL EXTENSION cl_khr_fp16 : enable", "convert_half2((float2)({v}))"),
}
_CHAINS, _UNROLL = 8, 4


def supported(dev, precision: str) -> bool:
    return {"FP32": True, "FP64": dev.supports_fp64, "FP16": dev.supports_fp16}[precision]


def _source(precision):
    t, _l, pragma, cast = _TYPES[precision]
    c = lambda v: cast.format(t=t, v=v)  # noqa: E731
    init = "\n".join(f"    {t} x{i} = {c(f'seed + {0.1 * i:.2f}f')};" for i in range(_CHAINS))
    body = "\n".join(f"        x{i} = fma(x{i}, a, b);" for _ in range(_UNROLL) for i in range(_CHAINS))
    total = " + ".join(f"x{i}" for i in range(_CHAINS))
    return f"""{pragma}
__kernel void peak(__global {t}* out, float fa, float fb, int iters)
{{
    const {t} a = {c('fa')};
    const {t} b = {c('fb')};
    const float seed = (float)get_global_id(0) * 1e-7f;
{init}
    for (int i = 0; i < iters; i++) {{
{body}
    }}
    out[get_global_id(0)] = {total};
}}
"""


def fma_peak(dev, precision: str, seconds: float, progress=None, cancel=None) -> float:
    """TFLOPS crête mesurés avec un noyau de FMA indépendantes."""
    t, lanes, _p, _c = _TYPES[precision]
    if not supported(dev, precision):
        raise opencl.OpenCLError(f"{precision} non exposé par le pilote OpenCL")
    elem = {"float": 4, "double": 8, "half2": 4}[t]
    local = min(256, dev.max_wg or 256)
    gsize = max(dev.compute_units, 1) * 4096
    gsize -= gsize % local
    per_iter = _CHAINS * _UNROLL * 2 * lanes
    with opencl.Session(dev) as s:
        k = s.kernel(s.build(_source(precision), "-cl-mad-enable"), "peak")
        s.set_arg(k, 0, s.buffer(gsize * elem))
        s.set_arg(k, 1, ctypes.c_float(0.999999))
        s.set_arg(k, 2, ctypes.c_float(1e-6))

        def launch(iters):
            s.set_arg(k, 3, ctypes.c_int32(iters))
            t0 = time.perf_counter()
            s.run(k, gsize, local)
            s.finish()
            return time.perf_counter() - t0

        iters = 16
        launch(iters)
        dt = launch(iters)
        while dt < 0.03 and iters < 1 << 22:  # calibration, < 0,25 s par lancement (TDR)
            if cancel and cancel():
                raise InterruptedError("annulé")
            iters *= 4 if dt < 0.005 else 2
            dt = launch(iters)
        iters = max(1, int(iters * min(0.2 / dt, 4)))
        best, start, runs = 0.0, time.perf_counter(), 0
        while (runs < 3 or time.perf_counter() - start < seconds) and runs < 300:
            if cancel and cancel():
                raise InterruptedError("annulé")
            best = max(best, gsize * iters * per_iter / launch(iters) / 1e12)
            runs += 1
            if progress:
                progress(min(1.0, (time.perf_counter() - start) / seconds))
        return best


def vram_bandwidth(dev, seconds: float, size_mb: int = 512, cancel=None) -> float:
    nbytes = min(size_mb * 1024 * 1024, int(dev.memory * 0.2))
    nbytes -= nbytes % 4096
    with opencl.Session(dev) as s:
        src, dst = s.buffer(nbytes, 1), s.buffer(nbytes, 1)  # CL_MEM_READ_WRITE
        s.copy(src, dst, nbytes); s.finish()
        t0 = time.perf_counter(); s.copy(src, dst, nbytes); s.finish()
        k = max(1, min(500, int(0.1 / max(time.perf_counter() - t0, 1e-6))))
        best, start, rounds = 0.0, time.perf_counter(), 0
        while rounds < 3 or time.perf_counter() - start < seconds:
            if cancel and cancel():
                raise InterruptedError("annulé")
            t0 = time.perf_counter()
            for _ in range(k):
                s.copy(src, dst, nbytes)
            s.finish()
            best = max(best, 2 * nbytes * k / (time.perf_counter() - t0) / 1e9)
            rounds += 1
        return best
