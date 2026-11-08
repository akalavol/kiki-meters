"""Accès au module natif C++ (tflops_native.dll) : calcul CPU et mémoire."""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import c_double, c_int

from .paths import resource_dir


class NativeUnavailable(RuntimeError):
    pass


def _load():
    name = "tflops_native.dll" if sys.platform == "win32" else "libtflops_native.so"
    path = os.path.join(resource_dir(), "bin", name)
    if not os.path.exists(path):
        return None, f"{name} introuvable (lancez build_native.cmd)"
    try:
        lib = ctypes.CDLL(path)
    except OSError as e:
        return None, f"Chargement de {name} impossible : {e}"
    lib.tm_version.restype = c_int
    lib.tm_cancel.argtypes = [c_int]
    lib.tm_cpu_features.restype = c_int
    lib.tm_cpu_flops.argtypes = [c_int, c_int, c_double]
    lib.tm_cpu_flops.restype = c_double
    lib.tm_mem_bandwidth.argtypes = [c_int, c_double, c_int, c_double]
    lib.tm_mem_bandwidth.restype = c_double
    lib.tm_mem_latency.argtypes = [c_double, c_double]
    lib.tm_mem_latency.restype = c_double
    if sys.platform == "win32" and lib.tm_version() >= 4:
        lib.tm_disk.argtypes = [ctypes.c_wchar_p, c_double, c_int, c_int, c_int, c_double]
        lib.tm_disk.restype = c_double
        lib.tm_disk_delete.argtypes = [ctypes.c_wchar_p]
        lib.tm_disk_delete.restype = c_int
        lib.tm_last_error.restype = ctypes.c_ulong
    return lib, ""


_lib, load_error = _load()


def available() -> bool:
    return _lib is not None


def _need():
    if _lib is None:
        raise NativeUnavailable(load_error)
    return _lib


def cancel():
    if _lib is not None:
        _lib.tm_cancel(1)


def cpu_features() -> dict:
    f = _need().tm_cpu_features()
    return {"avx2": bool(f & 1), "fma": bool(f & 2), "avx512f": bool(f & 4)}


def _checked(v: float) -> float:
    if v < 0:
        raise InterruptedError("mesure annulée ou mémoire insuffisante")
    return v


def cpu_gflops(precision: str, threads: int, seconds: float) -> float:
    """precision 'FP32' ou 'FP64' ; threads=0 → tous les threads logiques."""
    return _checked(_need().tm_cpu_flops(1 if precision == "FP64" else 0, threads, seconds))


MEM_MODES = {"read": 0, "write": 1, "copy": 2}


def mem_bandwidth(mode: str, size_mb: float, threads: int, seconds: float) -> float:
    """Débit en Go/s."""
    return _checked(_need().tm_mem_bandwidth(MEM_MODES[mode], size_mb, threads, seconds))


def mem_latency(size_mb: float, seconds: float) -> float:
    """Latence d'accès aléatoire en ns."""
    return _checked(_need().tm_mem_latency(size_mb, seconds))


DISK_MODES = {"prepare": 0, "seq_write": 1, "seq_read": 2, "rnd_read": 3, "rnd_write": 4}


def disk(path: str, mode: str, size_mb: float, block_kb: int, threads: int, seconds: float) -> float:
    """Débit disque en Mo/s (E/S directes, sans cache Windows)."""
    lib = _need()
    if not hasattr(lib, "tm_disk") or lib.tm_version() < 4:
        raise NativeUnavailable("module natif trop ancien : relancez build_native.cmd")
    v = lib.tm_disk(path, size_mb, DISK_MODES[mode], block_kb, threads, seconds)
    if v == -1:
        raise InterruptedError("mesure annulée")
    if v < 0:
        err = lib.tm_last_error()
        raise OSError(err, ctypes.FormatError(err).strip() if err else "erreur d'entrée/sortie", path)
    return v


def disk_delete(path: str):
    if _lib is not None and hasattr(_lib, "tm_disk_delete"):
        _lib.tm_disk_delete(path)
