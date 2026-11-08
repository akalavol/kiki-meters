"""NVIDIA : pilote CUDA (nvcuda.dll) + cuBLAS via ctypes.

Le pilote CUDA est installé avec tout pilote NVIDIA. cuBLAS vient du paquet pip
`nvidia-cublas-cu12` (ou d'un CUDA Toolkit présent sur la machine, ou du bundle).
Aucune compilation CUDA n'est nécessaire.
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys
import time
from ctypes import POINTER, byref, c_char_p, c_double, c_float, c_int, c_size_t, c_uint, c_uint64, c_void_p

from .paths import resource_dir


class CudaError(RuntimeError):
    pass


# --------------------------------------------------------------------------- pilote

def _load_driver():
    names = ["nvcuda.dll"] if sys.platform == "win32" else ["libcuda.so.1", "libcuda.so"]
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            pass
    return None


_drv = _load_driver()

CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

if _drv is not None:
    _drv.cuInit.argtypes = [c_uint]
    _drv.cuDeviceGetCount.argtypes = [POINTER(c_int)]
    _drv.cuDeviceGet.argtypes = [POINTER(c_int), c_int]
    _drv.cuDeviceGetName.argtypes = [c_char_p, c_int, c_int]
    _drv.cuDeviceGetAttribute.argtypes = [POINTER(c_int), c_int, c_int]
    _drv.cuDeviceTotalMem_v2.argtypes = [POINTER(c_size_t), c_int]
    _drv.cuDevicePrimaryCtxRetain.argtypes = [POINTER(c_void_p), c_int]
    _drv.cuDevicePrimaryCtxRelease_v2.argtypes = [c_int]
    _drv.cuCtxSetCurrent.argtypes = [c_void_p]
    _drv.cuCtxSynchronize.argtypes = []
    _drv.cuMemAlloc_v2.argtypes = [POINTER(c_uint64), c_size_t]
    _drv.cuMemFree_v2.argtypes = [c_uint64]
    _drv.cuMemsetD32_v2.argtypes = [c_uint64, c_uint, c_size_t]
    _drv.cuMemcpyDtoD_v2.argtypes = [c_uint64, c_uint64, c_size_t]
    _drv.cuGetErrorString.argtypes = [c_int, POINTER(c_char_p)]
    _drv.cuModuleLoadDataEx.argtypes = [POINTER(c_void_p), c_char_p, c_uint, POINTER(c_int), POINTER(c_void_p)]
    _drv.cuModuleGetFunction.argtypes = [POINTER(c_void_p), c_void_p, c_char_p]
    _drv.cuModuleUnload.argtypes = [c_void_p]
    _drv.cuLaunchKernel.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
                                    c_void_p, POINTER(c_void_p), POINTER(c_void_p)]


def _ck(rc, what):
    if rc != 0:
        msg = c_char_p()
        if _drv is not None:
            _drv.cuGetErrorString(rc, byref(msg))
        raise CudaError(f"{what} : {msg.value.decode() if msg.value else rc}")


_initialized = False


def driver_available() -> bool:
    global _initialized
    if _drv is None:
        return False
    if not _initialized:
        _initialized = _drv.cuInit(0) == 0
    return _initialized


class CudaDevice:
    def __init__(self, ordinal: int):
        self.ordinal = ordinal
        dev = c_int()
        _ck(_drv.cuDeviceGet(byref(dev), ordinal), "cuDeviceGet")
        self.handle = dev.value
        buf = ctypes.create_string_buffer(256)
        _drv.cuDeviceGetName(buf, 256, self.handle)
        self.name = buf.value.decode(errors="replace")
        self.sm_count = self._attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        self.clock_mhz = self._attr(CU_DEVICE_ATTRIBUTE_CLOCK_RATE) // 1000
        self.mem_clock_mhz = self._attr(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) // 1000
        self.bus_width = self._attr(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
        self.integrated = bool(self._attr(18))  # CU_DEVICE_ATTRIBUTE_INTEGRATED (Jetson, etc.)
        self.cc = (self._attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
                   self._attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR))
        mem = c_size_t()
        _drv.cuDeviceTotalMem_v2(byref(mem), self.handle)
        self.memory = mem.value

    def _attr(self, a):
        v = c_int()
        _drv.cuDeviceGetAttribute(byref(v), a, self.handle)
        return v.value

    @property
    def cc_str(self):
        return f"{self.cc[0]}.{self.cc[1]}"

    def __repr__(self):
        return f"<CudaDevice {self.ordinal} {self.name} sm_{self.cc[0]}{self.cc[1]}>"


def devices() -> list[CudaDevice]:
    if not driver_available():
        return []
    n = c_int()
    if _drv.cuDeviceGetCount(byref(n)) != 0:
        return []
    return [CudaDevice(i) for i in range(n.value)]


class Context:
    """Contexte primaire rendu courant dans le thread appelant."""

    def __init__(self, dev: CudaDevice):
        self.dev = dev
        ctx = c_void_p()
        _ck(_drv.cuDevicePrimaryCtxRetain(byref(ctx), dev.handle), "cuDevicePrimaryCtxRetain")
        self.ctx = ctx
        _ck(_drv.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        self._allocs = []
        self._modules = []

    def alloc(self, nbytes: int) -> int:
        p = c_uint64()
        _ck(_drv.cuMemAlloc_v2(byref(p), nbytes), f"allocation de {nbytes / 1e6:.0f} Mo")
        self._allocs.append(p.value)
        _ck(_drv.cuMemsetD32_v2(p.value, 0x3C003C00, nbytes // 4), "cuMemsetD32")  # 1.0 en FP16
        return p.value

    def free(self, ptr: int):
        if ptr in self._allocs:
            self._allocs.remove(ptr)
            _drv.cuMemFree_v2(ptr)

    def sync(self):
        _ck(_drv.cuCtxSynchronize(), "cuCtxSynchronize")

    def copy(self, dst: int, src: int, nbytes: int):
        _ck(_drv.cuMemcpyDtoD_v2(dst, src, nbytes), "cuMemcpyDtoD")

    def load_ptx(self, ptx: str):
        log = ctypes.create_string_buffer(8192)
        opts = (c_int * 2)(5, 6)  # CU_JIT_ERROR_LOG_BUFFER, ..._SIZE_BYTES
        vals = (c_void_p * 2)(ctypes.cast(log, c_void_p), c_void_p(8192))
        mod = c_void_p()
        rc = _drv.cuModuleLoadDataEx(byref(mod), ptx.encode() + b"\0", 2, opts, vals)
        if rc != 0:
            raise CudaError("compilation PTX : " + log.value.decode(errors="replace"))
        self._modules.append(mod)
        return mod

    @staticmethod
    def function(mod, name: str):
        f = c_void_p()
        _ck(_drv.cuModuleGetFunction(byref(f), mod, name.encode()), "cuModuleGetFunction")
        return f

    @staticmethod
    def launch(fn, grid: int, block: int, args: list):
        holders = [a if isinstance(a, ctypes._SimpleCData) else c_uint64(a) for a in args]
        params = (c_void_p * len(holders))(*[ctypes.cast(byref(h), c_void_p) for h in holders])
        _ck(_drv.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, None, params, None), "cuLaunchKernel")

    def close(self):
        try:
            _drv.cuCtxSynchronize()
        finally:
            for p in self._allocs:
                _drv.cuMemFree_v2(p)
            self._allocs.clear()
            for m in self._modules:
                _drv.cuModuleUnload(m)
            self._modules.clear()
            _drv.cuDevicePrimaryCtxRelease_v2(self.dev.handle)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- cuBLAS

def _cublas_dirs():
    dirs = []
    bundled = os.path.join(resource_dir(), "bin", "cuda")
    dirs.append(bundled)
    try:
        import nvidia.cublas as nc  # paquet pip nvidia-cublas-cu12
        for p in nc.__path__:
            dirs.append(os.path.join(p, "bin"))
            dirs.append(os.path.join(p, "lib"))
    except ImportError:
        pass
    try:
        import nvidia.cuda_runtime as nr
        for p in nr.__path__:
            dirs.append(os.path.join(p, "bin"))
    except ImportError:
        pass
    for var in ("CUDA_PATH",) + tuple(k for k in os.environ if k.startswith("CUDA_PATH_V")):
        if os.environ.get(var):
            dirs.append(os.path.join(os.environ[var], "bin"))
            dirs.append(os.path.join(os.environ[var], "bin", "x64"))
    return [d for d in dirs if os.path.isdir(d)]


def _load_cublas():
    if sys.platform != "win32":
        for n in ("libcublas.so.12", "libcublas.so"):
            try:
                return ctypes.CDLL(n), n
            except OSError:
                pass
        return None, "libcublas introuvable"
    for d in _cublas_dirs():
        os.add_dll_directory(d)
    for d in _cublas_dirs():
        for path in sorted(glob.glob(os.path.join(d, "cublas64_*.dll")), reverse=True):
            try:
                return ctypes.CDLL(path), path
            except OSError as e:
                last = f"{path} : {e}"
                continue
    return None, "cuBLAS introuvable (pip install nvidia-cublas-cu12)"


_cublas = None
cublas_info = ""


def cublas():
    global _cublas, cublas_info
    if _cublas is None:
        _cublas, cublas_info = _load_cublas()
        if _cublas is not None:
            _cublas.cublasCreate_v2.argtypes = [POINTER(c_void_p)]
            _cublas.cublasDestroy_v2.argtypes = [c_void_p]
            _cublas.cublasSetMathMode.argtypes = [c_void_p, c_int]
            _cublas.cublasGetVersion_v2.argtypes = [c_void_p, POINTER(c_int)]
            _cublas.cublasGemmEx.argtypes = [
                c_void_p, c_int, c_int, c_int, c_int, c_int,
                c_void_p, c_uint64, c_int, c_int, c_uint64, c_int, c_int,
                c_void_p, c_uint64, c_int, c_int, c_int, c_int]
    return _cublas


def cublas_available() -> bool:
    return cublas() is not None


# Types cudaDataType / cublasComputeType
CUDA_R_16F, CUDA_R_32F, CUDA_R_64F, CUDA_R_16BF = 2, 0, 1, 14
CUBLAS_COMPUTE_16F, CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_COMPUTE_64F = 64, 68, 77, 70
CUBLAS_DEFAULT_MATH, CUBLAS_PEDANTIC_MATH, CUBLAS_TF32_TENSOR_OP_MATH = 0, 2, 3
CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99

# précision → (type données, octets, type de calcul, type des scalaires, cc minimal, mode)
GEMM_PRECISIONS = {
    "FP64": (CUDA_R_64F, 8, CUBLAS_COMPUTE_64F, "d", (1, 0), CUBLAS_DEFAULT_MATH),
    "FP32": (CUDA_R_32F, 4, CUBLAS_COMPUTE_32F, "f", (1, 0), CUBLAS_PEDANTIC_MATH),
    "TF32": (CUDA_R_32F, 4, CUBLAS_COMPUTE_32F_FAST_TF32, "f", (8, 0), CUBLAS_TF32_TENSOR_OP_MATH),
    "FP16": (CUDA_R_16F, 2, CUBLAS_COMPUTE_32F, "f", (5, 3), CUBLAS_DEFAULT_MATH),
    "BF16": (CUDA_R_16BF, 2, CUBLAS_COMPUTE_32F, "f", (8, 0), CUBLAS_DEFAULT_MATH),
}


def gemm_supported(dev: CudaDevice, precision: str) -> bool:
    return dev.cc >= GEMM_PRECISIONS[precision][4]


def gemm_tflops(dev: CudaDevice, precision: str, seconds: float, n: int | None = None,
                progress=None, cancel=None) -> tuple[float, int]:
    """GEMM carré n×n via cuBLAS ; renvoie (meilleurs TFLOPS, n)."""
    lib = cublas()
    if lib is None:
        raise CudaError(cublas_info)
    dtype, size, compute, scalar, _cc, math_mode = GEMM_PRECISIONS[precision]
    if n is None:
        n = 8192 if size <= 4 else 4096
        # au plus ~40 % de la VRAM pour 3 matrices
        while 3 * n * n * size > dev.memory * 0.4 and n > 1024:
            n //= 2
    with Context(dev) as ctx:
        a, b, c = (ctx.alloc(n * n * size) for _ in range(3))
        handle = c_void_p()
        rc = lib.cublasCreate_v2(byref(handle))
        if rc != 0:
            raise CudaError(f"cublasCreate : code {rc}")
        try:
            lib.cublasSetMathMode(handle, math_mode)
            one, zero = (c_double(1.0), c_double(0.0)) if scalar == "d" else (c_float(1.0), c_float(0.0))

            def gemm():
                rc = lib.cublasGemmEx(handle, 0, 0, n, n, n, ctypes.cast(byref(one), c_void_p),
                                      a, dtype, n, b, dtype, n, ctypes.cast(byref(zero), c_void_p),
                                      c, dtype, n, compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
                if rc != 0:
                    raise CudaError(f"cublasGemmEx {precision} : code {rc}")

            flop = 2.0 * n ** 3
            gemm(); ctx.sync()  # choix de l'algorithme + montée en fréquence
            # Lots de k GEMM d'environ 0,15 s chacun (sous la limite TDR de Windows).
            t0 = time.perf_counter(); gemm(); ctx.sync()
            k = max(1, min(200, int(0.15 / max(time.perf_counter() - t0, 1e-5))))
            best, start = 0.0, time.perf_counter()
            batches = 0
            while batches < 3 or time.perf_counter() - start < seconds:
                if cancel and cancel():
                    raise InterruptedError("annulé")
                t0 = time.perf_counter()
                for _ in range(k):
                    gemm()
                ctx.sync()
                best = max(best, flop * k / (time.perf_counter() - t0) / 1e12)
                batches += 1
                if progress:
                    progress(min(1.0, (time.perf_counter() - start) / seconds))
                if batches > 1000:
                    break
            return best, n
        finally:
            lib.cublasDestroy_v2(handle)


def gemm_stream(dev: CudaDevice, precision: str, n: int):
    """Générateur pour le test de stabilité : chaque next() renvoie les TFLOPS d'un lot (~0,25 s)."""
    lib = cublas()
    if lib is None:
        raise CudaError(cublas_info)
    dtype, size, compute, scalar, _cc, math_mode = GEMM_PRECISIONS[precision]
    ctx = Context(dev)
    handle = c_void_p()
    try:
        a, b, c = (ctx.alloc(n * n * size) for _ in range(3))
        if lib.cublasCreate_v2(byref(handle)) != 0:
            raise CudaError("cublasCreate")
        lib.cublasSetMathMode(handle, math_mode)
        one, zero = (c_double(1.0), c_double(0.0)) if scalar == "d" else (c_float(1.0), c_float(0.0))

        def gemm():
            rc = lib.cublasGemmEx(handle, 0, 0, n, n, n, ctypes.cast(byref(one), c_void_p),
                                  a, dtype, n, b, dtype, n, ctypes.cast(byref(zero), c_void_p),
                                  c, dtype, n, compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
            if rc != 0:
                raise CudaError(f"cublasGemmEx : code {rc}")

        gemm(); ctx.sync()
        t0 = time.perf_counter(); gemm(); ctx.sync()
        k = max(1, min(400, int(0.25 / max(time.perf_counter() - t0, 1e-5))))
        while True:
            t0 = time.perf_counter()
            for _ in range(k):
                gemm()
            ctx.sync()
            yield 2.0 * n ** 3 * k / (time.perf_counter() - t0) / 1e12
    finally:
        if handle:
            lib.cublasDestroy_v2(handle)
        ctx.close()


# --------------------------------------------------------------------------- VRAM

def vram_bandwidth(dev: CudaDevice, seconds: float, size_mb: int = 1024, cancel=None) -> float:
    """Copie mémoire GPU → GPU ; Go/s = (lecture + écriture) / temps."""
    nbytes = size_mb * 1024 * 1024
    nbytes = min(nbytes, int(dev.memory * 0.2))
    nbytes -= nbytes % 4096
    with Context(dev) as ctx:
        src, dst = ctx.alloc(nbytes), ctx.alloc(nbytes)
        ctx.copy(dst, src, nbytes); ctx.sync()
        t0 = time.perf_counter(); ctx.copy(dst, src, nbytes); ctx.sync()
        k = max(1, min(500, int(0.1 / max(time.perf_counter() - t0, 1e-6))))
        best, start, rounds = 0.0, time.perf_counter(), 0
        while rounds < 3 or time.perf_counter() - start < seconds:
            if cancel and cancel():
                raise InterruptedError("annulé")
            t0 = time.perf_counter()
            for _ in range(k):
                ctx.copy(dst, src, nbytes)
            ctx.sync()
            best = max(best, 2 * nbytes * k / (time.perf_counter() - t0) / 1e9)
            rounds += 1
        return best


def theoretical_vram_gbs(dev: CudaDevice) -> float | None:
    """Débit constructeur approximatif. L'attribut d'horloge mémoire n'est pas
    toujours renseigné (GDDR7) ; on renvoie alors None."""
    if not dev.mem_clock_mhz or not dev.bus_width:
        return None
    return dev.mem_clock_mhz * 1e6 * 2 * dev.bus_width / 8 / 1e9
