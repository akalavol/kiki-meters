"""Surveillance matérielle : NVML (GPU NVIDIA) et compteurs de performance Windows (CPU)."""
from __future__ import annotations

import ctypes
import sys
from ctypes import POINTER, byref, c_char_p, c_double, c_uint, c_ulong, c_ulonglong, c_void_p, c_wchar_p

# --------------------------------------------------------------------------- NVML

THROTTLE_REASONS = {
    0x4: "limite de puissance",
    0x8: "ralentissement matériel",
    0x20: "limite thermique (logiciel)",
    0x40: "limite thermique (matériel)",
    0x80: "frein d'alimentation",
}


class _Util(ctypes.Structure):
    _fields_ = [("gpu", c_uint), ("memory", c_uint)]


class _Mem(ctypes.Structure):
    _fields_ = [("total", c_ulonglong), ("free", c_ulonglong), ("used", c_ulonglong)]


class Nvml:
    def __init__(self):
        self.lib = None
        self.handles = []
        self.names = []
        if sys.platform == "win32":
            candidates = ["nvml.dll", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"]
        else:
            candidates = ["libnvidia-ml.so.1"]
        for c in candidates:
            try:
                self.lib = ctypes.CDLL(c)
                break
            except OSError:
                continue
        if self.lib is None or self.lib.nvmlInit_v2() != 0:
            self.lib = None
            return
        n = c_uint()
        self.lib.nvmlDeviceGetCount_v2(byref(n))
        for i in range(n.value):
            h = c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(i, byref(h)) == 0:
                buf = ctypes.create_string_buffer(96)
                self.lib.nvmlDeviceGetName(h, buf, 96)
                self.handles.append(h)
                self.names.append(buf.value.decode(errors="replace"))
        self._reasons_fn = getattr(self.lib, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            getattr(self.lib, "nvmlDeviceGetCurrentClocksThrottleReasons", None)

    @property
    def available(self):
        return self.lib is not None and bool(self.handles)

    def driver_version(self):
        if not self.lib:
            return None
        buf = ctypes.create_string_buffer(80)
        return buf.value.decode() if self.lib.nvmlSystemGetDriverVersion(buf, 80) == 0 else None

    def _uint(self, fn, h, *args):
        v = c_uint()
        return v.value if getattr(self.lib, fn)(h, *args, byref(v)) == 0 else None

    def sample(self, index=0) -> dict | None:
        if not self.available or index >= len(self.handles):
            return None
        h = self.handles[index]
        util, mem = _Util(), _Mem()
        power = self._uint("nvmlDeviceGetPowerUsage", h)
        limit = self._uint("nvmlDeviceGetEnforcedPowerLimit", h)
        reasons = None
        if self._reasons_fn:
            r = c_ulonglong()
            if self._reasons_fn(h, byref(r)) == 0:
                reasons = r.value
        return {
            "name": self.names[index],
            "temp_c": self._uint("nvmlDeviceGetTemperature", h, 0),
            "clock_mhz": self._uint("nvmlDeviceGetClockInfo", h, 1),     # SM
            "mem_clock_mhz": self._uint("nvmlDeviceGetClockInfo", h, 2),
            "power_w": power / 1000 if power is not None else None,
            "power_limit_w": limit / 1000 if limit is not None else None,
            "util_pct": util.gpu if self.lib.nvmlDeviceGetUtilizationRates(h, byref(util)) == 0 else None,
            "fan_pct": self._uint("nvmlDeviceGetFanSpeed", h),
            "mem_used_gb": mem.used / 1024 ** 3 if self.lib.nvmlDeviceGetMemoryInfo(h, byref(mem)) == 0 else None,
            "throttle": [txt for bit, txt in THROTTLE_REASONS.items() if reasons and reasons & bit],
        }

    def index_for(self, name: str) -> int | None:
        for i, n in enumerate(self.names):
            if n == name:
                return i
        return None


# --------------------------------------------------------------------------- CPU (PDH)

class _PdhValue(ctypes.Structure):
    _fields_ = [("CStatus", c_ulong), ("doubleValue", c_double)]


class CpuCounters:
    """Utilisation et fréquence effective : % Processor Performance × Processor Frequency.
    (La température CPU n'est pas exposée par Windows sans pilote tiers.)"""

    def __init__(self):
        self.ok = False
        if sys.platform != "win32":
            return
        try:
            self.pdh = ctypes.WinDLL("pdh.dll")
        except OSError:
            return
        self.pdh.PdhOpenQueryW.argtypes = [c_wchar_p, c_void_p, POINTER(c_void_p)]
        self.pdh.PdhAddEnglishCounterW.argtypes = [c_void_p, c_wchar_p, c_void_p, POINTER(c_void_p)]
        self.pdh.PdhCollectQueryData.argtypes = [c_void_p]
        self.pdh.PdhGetFormattedCounterValue.argtypes = [c_void_p, c_ulong, c_void_p, POINTER(_PdhValue)]
        self.query = c_void_p()
        if self.pdh.PdhOpenQueryW(None, None, byref(self.query)) != 0:
            return
        self.counters = {}
        for key, path in {
            "usage": r"\Processor Information(_Total)\% Processor Utility",
            "usage_alt": r"\Processor(_Total)\% Processor Time",
            "perf": r"\Processor Information(_Total)\% Processor Performance",
            "freq": r"\Processor Information(_Total)\Processor Frequency",
        }.items():
            c = c_void_p()
            if self.pdh.PdhAddEnglishCounterW(self.query, path, None, byref(c)) == 0:
                self.counters[key] = c
        self.pdh.PdhCollectQueryData(self.query)
        self.ok = True

    def _get(self, key):
        c = self.counters.get(key)
        if c is None:
            return None
        v = _PdhValue()
        if self.pdh.PdhGetFormattedCounterValue(c, 0x200 | 0x8000, None, byref(v)) != 0:  # DOUBLE | NOCAP100
            return None
        return v.doubleValue

    def sample(self) -> dict | None:
        if not self.ok:
            return None
        self.pdh.PdhCollectQueryData(self.query)
        usage = self._get("usage")
        if usage is None:
            usage = self._get("usage_alt")
        perf, freq = self._get("perf"), self._get("freq")
        return {
            "usage_pct": min(usage, 100.0) if usage is not None else None,
            "mhz": perf * freq / 100 if perf and freq else freq,
        }


class Monitor:
    def __init__(self):
        self.nvml = Nvml()
        self.cpu = CpuCounters()

    def sample(self) -> dict:
        return {
            "cpu": self.cpu.sample(),
            "gpus": [self.nvml.sample(i) for i in range(len(self.nvml.handles))],
        }
