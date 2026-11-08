"""Liaison OpenCL minimale via ctypes (aucune dépendance externe).

OpenCL est fourni par les pilotes NVIDIA, AMD et Intel : la DLL suffit,
pas besoin du CUDA Toolkit ni de PyTorch.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import sys
from ctypes import POINTER, byref, c_char_p, c_int32, c_size_t, c_uint32, c_uint64, c_void_p

CL_SUCCESS = 0
CL_DEVICE_TYPE_CPU = 1 << 1
CL_DEVICE_TYPE_GPU = 1 << 2
CL_DEVICE_TYPE_ACCELERATOR = 1 << 3
CL_DEVICE_TYPE_ALL = 0xFFFFFFFF

CL_DEVICE_TYPE = 0x1000
CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002
CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004
CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100C
CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F
CL_DEVICE_NAME = 0x102B
CL_DEVICE_VENDOR = 0x102C
CL_DRIVER_VERSION = 0x102D
CL_DEVICE_VERSION = 0x102F
CL_DEVICE_EXTENSIONS = 0x1030
CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV = 0x4000
CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV = 0x4001

CL_MEM_WRITE_ONLY = 1 << 1
CL_PROGRAM_BUILD_LOG = 0x1183


class OpenCLError(RuntimeError):
    pass


def _load_library():
    candidates = []
    if sys.platform == "win32":
        candidates = ["OpenCL.dll"]
    elif sys.platform == "darwin":
        candidates = ["/System/Library/Frameworks/OpenCL.framework/OpenCL"]
    else:
        candidates = ["libOpenCL.so.1", "libOpenCL.so"]
    found = ctypes.util.find_library("OpenCL")
    if found:
        candidates.append(found)
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


_lib = _load_library()


def available() -> bool:
    return _lib is not None


def _check(err, what):
    code = err.value if hasattr(err, "value") else err
    if code != CL_SUCCESS:
        raise OpenCLError(f"{what} a échoué (code OpenCL {code})")


if _lib is not None:
    _lib.clGetPlatformIDs.argtypes = [c_uint32, POINTER(c_void_p), POINTER(c_uint32)]
    _lib.clGetDeviceIDs.argtypes = [c_void_p, c_uint64, c_uint32, POINTER(c_void_p), POINTER(c_uint32)]
    _lib.clGetDeviceInfo.argtypes = [c_void_p, c_uint32, c_size_t, c_void_p, POINTER(c_size_t)]
    _lib.clCreateContext.argtypes = [c_void_p, c_uint32, POINTER(c_void_p), c_void_p, c_void_p, POINTER(c_int32)]
    _lib.clCreateContext.restype = c_void_p
    _lib.clCreateCommandQueue.argtypes = [c_void_p, c_void_p, c_uint64, POINTER(c_int32)]
    _lib.clCreateCommandQueue.restype = c_void_p
    _lib.clCreateProgramWithSource.argtypes = [c_void_p, c_uint32, POINTER(c_char_p), POINTER(c_size_t), POINTER(c_int32)]
    _lib.clCreateProgramWithSource.restype = c_void_p
    _lib.clBuildProgram.argtypes = [c_void_p, c_uint32, POINTER(c_void_p), c_char_p, c_void_p, c_void_p]
    _lib.clGetProgramBuildInfo.argtypes = [c_void_p, c_void_p, c_uint32, c_size_t, c_void_p, POINTER(c_size_t)]
    _lib.clCreateKernel.argtypes = [c_void_p, c_char_p, POINTER(c_int32)]
    _lib.clCreateKernel.restype = c_void_p
    _lib.clCreateBuffer.argtypes = [c_void_p, c_uint64, c_size_t, c_void_p, POINTER(c_int32)]
    _lib.clCreateBuffer.restype = c_void_p
    _lib.clSetKernelArg.argtypes = [c_void_p, c_uint32, c_size_t, c_void_p]
    _lib.clEnqueueNDRangeKernel.argtypes = [c_void_p, c_void_p, c_uint32, POINTER(c_size_t), POINTER(c_size_t),
                                            POINTER(c_size_t), c_uint32, c_void_p, c_void_p]
    _lib.clFinish.argtypes = [c_void_p]
    _lib.clEnqueueCopyBuffer.argtypes = [c_void_p, c_void_p, c_void_p, c_size_t, c_size_t, c_size_t,
                                         c_uint32, c_void_p, c_void_p]
    for fn in ("clReleaseMemObject", "clReleaseKernel", "clReleaseProgram",
               "clReleaseCommandQueue", "clReleaseContext"):
        getattr(_lib, fn).argtypes = [c_void_p]


def _info_raw(device, param) -> bytes:
    size = c_size_t(0)
    _check(_lib.clGetDeviceInfo(device, param, 0, None, byref(size)), "clGetDeviceInfo")
    buf = ctypes.create_string_buffer(size.value)
    _check(_lib.clGetDeviceInfo(device, param, size.value, buf, None), "clGetDeviceInfo")
    return buf.raw


def _info_str(device, param) -> str:
    return _info_raw(device, param).rstrip(b"\x00").decode("utf-8", "replace").strip()


def _info_int(device, param, default=0) -> int:
    try:
        return int.from_bytes(_info_raw(device, param), "little")
    except OpenCLError:
        return default


class Device:
    def __init__(self, handle, platform):
        self.handle = handle
        self.platform = platform
        self.name = _info_str(handle, CL_DEVICE_NAME)
        self.vendor = _info_str(handle, CL_DEVICE_VENDOR)
        self.driver = _info_str(handle, CL_DRIVER_VERSION)
        self.version = _info_str(handle, CL_DEVICE_VERSION)
        self.type = _info_int(handle, CL_DEVICE_TYPE)
        self.compute_units = _info_int(handle, CL_DEVICE_MAX_COMPUTE_UNITS)
        self.clock_mhz = _info_int(handle, CL_DEVICE_MAX_CLOCK_FREQUENCY)
        self.max_wg = _info_int(handle, CL_DEVICE_MAX_WORK_GROUP_SIZE, 256)
        self.memory = _info_int(handle, CL_DEVICE_GLOBAL_MEM_SIZE)
        self.extensions = set(_info_str(handle, CL_DEVICE_EXTENSIONS).split())
        # Mémoire partagée avec le CPU = GPU intégré (sa « VRAM » est en réalité de la RAM)
        self.integrated = bool(_info_int(handle, 0x1035))  # CL_DEVICE_HOST_UNIFIED_MEMORY
        self.cc = None
        if "cl_nv_device_attribute_query" in self.extensions:
            self.cc = (_info_int(handle, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV),
                       _info_int(handle, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV))

    @property
    def is_gpu(self):
        return bool(self.type & CL_DEVICE_TYPE_GPU)

    @property
    def kind(self):
        if self.type & CL_DEVICE_TYPE_GPU:
            return "GPU"
        if self.type & CL_DEVICE_TYPE_CPU:
            return "CPU"
        return "Accélérateur"

    @property
    def supports_fp64(self):
        return "cl_khr_fp64" in self.extensions

    @property
    def supports_fp16(self):
        return "cl_khr_fp16" in self.extensions

    def __repr__(self):
        return f"<Device {self.name} ({self.kind})>"


def list_devices(device_type=CL_DEVICE_TYPE_ALL):
    if _lib is None:
        return []
    count = c_uint32(0)
    if _lib.clGetPlatformIDs(0, None, byref(count)) != CL_SUCCESS or count.value == 0:
        return []
    platforms = (c_void_p * count.value)()
    _check(_lib.clGetPlatformIDs(count.value, platforms, None), "clGetPlatformIDs")
    devices = []
    for plat in platforms:
        n = c_uint32(0)
        if _lib.clGetDeviceIDs(plat, device_type, 0, None, byref(n)) != CL_SUCCESS or n.value == 0:
            continue
        handles = (c_void_p * n.value)()
        if _lib.clGetDeviceIDs(plat, device_type, n.value, handles, None) != CL_SUCCESS:
            continue
        for h in handles:
            try:
                devices.append(Device(h, plat))
            except OpenCLError:
                pass
    return devices


class Session:
    """Contexte + file de commandes pour un périphérique."""

    def __init__(self, device: Device):
        self.device = device
        err = c_int32(0)
        dev = (c_void_p * 1)(device.handle)
        self.ctx = _lib.clCreateContext(None, 1, dev, None, None, byref(err))
        _check(err, "clCreateContext")
        self.queue = _lib.clCreateCommandQueue(self.ctx, device.handle, 0, byref(err))
        _check(err, "clCreateCommandQueue")
        self._objects = []

    def build(self, source: str, options: str = ""):
        err = c_int32(0)
        src = c_char_p(source.encode())
        length = c_size_t(len(source.encode()))
        prog = _lib.clCreateProgramWithSource(self.ctx, 1, byref(src), byref(length), byref(err))
        _check(err, "clCreateProgramWithSource")
        dev = (c_void_p * 1)(self.device.handle)
        rc = _lib.clBuildProgram(prog, 1, dev, options.encode(), None, None)
        if rc != CL_SUCCESS:
            size = c_size_t(0)
            _lib.clGetProgramBuildInfo(prog, self.device.handle, CL_PROGRAM_BUILD_LOG, 0, None, byref(size))
            log = ctypes.create_string_buffer(max(size.value, 1))
            _lib.clGetProgramBuildInfo(prog, self.device.handle, CL_PROGRAM_BUILD_LOG, size.value, log, None)
            _lib.clReleaseProgram(prog)
            raise OpenCLError("Compilation du noyau impossible :\n" + log.value.decode("utf-8", "replace"))
        self._objects.append(("clReleaseProgram", prog))
        return prog

    def kernel(self, program, name: str):
        err = c_int32(0)
        k = _lib.clCreateKernel(program, name.encode(), byref(err))
        _check(err, "clCreateKernel")
        self._objects.append(("clReleaseKernel", k))
        return k

    def copy(self, src, dst, nbytes: int):
        _check(_lib.clEnqueueCopyBuffer(self.queue, src, dst, 0, 0, nbytes, 0, None, None),
               "clEnqueueCopyBuffer")

    def buffer(self, nbytes: int, flags: int = CL_MEM_WRITE_ONLY):
        err = c_int32(0)
        b = _lib.clCreateBuffer(self.ctx, flags, nbytes, None, byref(err))
        _check(err, "clCreateBuffer")
        self._objects.append(("clReleaseMemObject", b))
        return b

    @staticmethod
    def set_arg(kernel, index: int, value):
        if isinstance(value, ctypes._SimpleCData):
            _check(_lib.clSetKernelArg(kernel, index, ctypes.sizeof(value), byref(value)), "clSetKernelArg")
        else:  # handle mémoire (c_void_p brut)
            h = c_void_p(value)
            _check(_lib.clSetKernelArg(kernel, index, ctypes.sizeof(h), byref(h)), "clSetKernelArg")

    def run(self, kernel, global_size: int, local_size: int | None):
        g = (c_size_t * 1)(global_size)
        l = (c_size_t * 1)(local_size) if local_size else None
        _check(_lib.clEnqueueNDRangeKernel(self.queue, kernel, 1, None, g, l, 0, None, None),
               "clEnqueueNDRangeKernel")

    def finish(self):
        _check(_lib.clFinish(self.queue), "clFinish")

    def close(self):
        for fn, obj in reversed(self._objects):
            getattr(_lib, fn)(obj)
        self._objects.clear()
        if self.queue:
            _lib.clReleaseCommandQueue(self.queue)
            self.queue = None
        if self.ctx:
            _lib.clReleaseContext(self.ctx)
            self.ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
