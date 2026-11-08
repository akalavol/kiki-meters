"""Description du système (CPU, RAM, GPU, bibliothèques disponibles)."""
from __future__ import annotations

import ctypes
import os
import platform

from . import cuda, gpu_opencl, native


def cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "Processeur inconnu"


def physical_cores() -> int | None:
    if platform.system() != "Windows":
        return None
    try:
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        size = wintypes.DWORD(0)
        k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))  # RelationProcessorCore
        buf = ctypes.create_string_buffer(size.value)
        if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
            return None
        count, offset = 0, 0
        while offset < size.value:
            offset += int.from_bytes(buf.raw[offset + 4:offset + 8], "little")
            count += 1
        return count
    except Exception:
        return None


def total_ram_gb() -> float | None:
    if platform.system() == "Windows":
        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(m)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return m.ullTotalPhys / 1024 ** 3
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (ValueError, OSError, AttributeError):
        return None


def collect() -> dict:
    cuda_devs = cuda.devices()
    cl_gpus = gpu_opencl.gpu_devices()
    gpus = []
    for d in cuda_devs:
        gpus.append({
            "name": d.name, "vendor": "NVIDIA", "api": "CUDA", "compute_capability": d.cc_str,
            "units": d.sm_count, "clock_mhz": d.clock_mhz, "memory_gb": d.memory / 1024 ** 3,
            "bus_width": d.bus_width, "integrated": d.integrated,
        })
    cuda_names = {d.name for d in cuda_devs}
    for g in cl_gpus:
        if g.name in cuda_names:
            continue
        gpus.append({
            "name": g.name, "vendor": g.vendor, "api": "OpenCL", "compute_capability": None,
            "units": g.compute_units, "clock_mhz": g.clock_mhz, "memory_gb": g.memory / 1024 ** 3,
            "bus_width": None, "integrated": g.integrated,
        })
    feats = native.cpu_features() if native.available() else {}
    return {
        "cpu": cpu_name(),
        "cores": physical_cores(),
        "threads": os.cpu_count(),
        "cpu_features": [k.upper() for k, v in feats.items() if v],
        "ram_gb": total_ram_gb(),
        "os": f"{platform.system()} {platform.release()} (build {platform.version()})",
        "gpus": gpus,
        "native": native.available(),
        "native_error": native.load_error,
        "cuda": cuda.driver_available(),
        "cublas": cuda.cublas_available(),
        "cublas_info": cuda.cublas_info,
    }
