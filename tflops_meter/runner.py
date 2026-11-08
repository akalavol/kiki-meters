"""Orchestration des tests : analyse complète et test de stabilité.

Les fonctions sont bloquantes ; l'interface les exécute dans un thread de travail et
reçoit l'avancement par des rappels (callbacks).
"""
from __future__ import annotations

import os
import statistics
import time
from typing import Callable

from . import cuda, gpu_opencl, hwdetails, native, opencl
from .model import PRESET_DISK_MB, PRESET_SECONDS, Measure, Settings

_temp_files: set[str] = set()  # fichiers de test disque à supprimer quoi qu'il arrive


def default_disk_letters(drives: list[dict]) -> list[str]:
    """Disques internes réels : ni amovibles, ni USB, ni lecteurs virtuels (Google Drive…)."""
    real = [d["letter"] for d in drives if not d.get("removable") and d.get("model") and d.get("kind") != "USB"]
    if real:
        return real
    system = (os.environ.get("SystemDrive") or "C:")[0]
    return [d["letter"] for d in drives if d["letter"] == system]


def _disk_tasks(cfg: Settings, secs: float, drives: list[dict]):
    tasks = []
    letters = cfg.disk_letters or default_disk_letters(drives)
    size = cfg.disk_size_mb or PRESET_DISK_MB.get(cfg.preset, 1024)
    for dr in drives:
        if dr["letter"] not in letters:
            continue
        label = hwdetails.drive_label(dr)
        path = hwdetails.test_path(dr)

        def first(_p, dr=dr, label=label, path=path):
            if dr["free_gb"] * 1000 < size * 1.1 + 1000:
                raise OSError(f"espace libre insuffisant sur {dr['letter']}: ({dr['free_gb']:.1f} Go)")
            _temp_files.add(path)
            native.disk(path, "prepare", size, 1024, 1, 0)  # remplit le fichier de données aléatoires
            v = native.disk(path, "seq_write", size, 1024, 8, secs)
            return Measure("Disque", label, "Écriture séquentielle", v, "Mo/s",
                           details=f"blocs de 1 Mo, 8 en parallèle, fichier de {size} Mo")

        def step(mode, test, block, threads, last=False, label=label, path=path):
            def run(_p):
                try:
                    if path not in _temp_files:
                        raise OSError("fichier de test absent (étape précédente en échec)")
                    v = native.disk(path, mode, size, block, threads, secs)
                    det = f"blocs de {f'{block // 1024} Mo' if block >= 1024 else f'{block} Ko'}, {threads} en parallèle"
                    if block == 4:
                        det += f" · ≈ {v * 1e6 / 4096:,.0f} IOPS".replace(",", " ")
                    return Measure("Disque", label, test, v, "Mo/s", details=det)
                finally:
                    if last:
                        _cleanup()
            return run

        tasks.append((f"{label} écriture séquentielle", first))
        tasks.append((f"{label} lecture séquentielle", step("seq_read", "Lecture séquentielle", 1024, 8)))
        tasks.append((f"{label} lecture 4K", step("rnd_read", "Lecture aléatoire 4K (Q1)", 4, 1)))
        tasks.append((f"{label} écriture 4K", step("rnd_write", "Écriture aléatoire 4K (Q1)", 4, 1)))
        tasks.append((f"{label} lecture 4K Q32", step("rnd_read", "Lecture aléatoire 4K (Q32)", 4, 32,
                                                      last=True)))
    return tasks


def _cleanup():
    for p in list(_temp_files):
        native.disk_delete(p)
        try:
            os.remove(p)
        except OSError:
            pass
        if not os.path.exists(p):
            _temp_files.discard(p)
from .monitor import Monitor, Nvml

GEMM_LABELS = {"FP32": "GEMM FP32", "TF32": "GEMM TF32 (Tensor)", "FP16": "GEMM FP16 (Tensor)",
               "BF16": "GEMM BF16 (Tensor)", "FP64": "GEMM FP64"}

_FP64_RATIO = {(5, 3): 1 / 32, (6, 0): 1 / 2, (6, 1): 1 / 32, (6, 2): 1 / 32, (7, 0): 1 / 2, (7, 2): 1 / 32,
               (7, 5): 1 / 32, (8, 0): 1 / 2, (9, 0): 1 / 2, (10, 0): 1 / 2}


def cuda_fp32_peak(dev: cuda.CudaDevice) -> float | None:
    lanes = gpu_opencl._NV_CORES_PER_SM.get(dev.cc) or (128 if dev.cc[0] >= 8 else None)
    if not lanes or not dev.clock_mhz:
        return None
    return dev.sm_count * lanes * 2 * dev.clock_mhz * 1e6 / 1e12


def cuda_theoretical(dev: cuda.CudaDevice, precision: str) -> float | None:
    fp32 = cuda_fp32_peak(dev)
    if fp32 is None:
        return None
    if precision == "FP32":
        return fp32
    if precision == "FP64":
        return fp32 * _FP64_RATIO.get(dev.cc, 1 / 64 if dev.cc[0] >= 8 else 1 / 32)
    return None  # Tensor Cores : dépend du modèle exact, pas d'estimation fiable


class Cancelled(Exception):
    pass


def _gpu_busy_warning() -> str | None:
    nv = Nvml()
    if not nv.available:
        return None
    # Plusieurs relevés sur ~1 s : on garde le minimum pour ignorer les pics ponctuels
    # (bureau Windows, navigateur…) et ne signaler qu'une charge continue.
    lowest: dict[int, tuple[str, int]] = {}
    for k in range(5):
        for i in range(len(nv.handles)):
            s = nv.sample(i)
            if s and s.get("util_pct") is not None:
                prev = lowest.get(i)
                if prev is None or s["util_pct"] < prev[1]:
                    lowest[i] = (s["name"], s["util_pct"])
        if k < 4:
            time.sleep(0.25)
    busy = [f"{name} ({util} %)" for name, util in lowest.values() if util >= 40]
    if busy:
        return ("GPU déjà occupé avant le test : " + ", ".join(busy) +
                ". Fermez les jeux, IA ou rendus en cours, sinon les résultats seront sous-estimés.")
    return None


def build_tasks(cfg: Settings, drives: list[dict] | None = None):
    """Liste de (libellé, fonction(progress) -> Measure)."""
    secs = PRESET_SECONDS.get(cfg.preset, 2.0)
    tasks: list[tuple[str, Callable]] = []

    if cfg.do_cpu:
        def cpu_task(prec, threads, label):
            def run(_progress):
                from .sysinfo import cpu_name
                if not native.available():
                    return Measure("CPU", cpu_name(), label, None, "GFLOPS", error=native.load_error)
                feats = native.cpu_features()
                isa = "AVX2 + FMA" if feats["avx2"] and feats["fma"] else "SSE2"
                v = native.cpu_gflops(prec, threads, secs)
                n = "1 thread" if threads == 1 else f"{threads or 'tous les'} threads"
                return Measure("CPU", cpu_name(), label, v, "GFLOPS", details=f"FMA {isa}, {n}")
            return run
        mt = cfg.cpu_threads or 0
        for prec in ("FP32", "FP64"):
            tasks.append((f"CPU {prec} monocœur", cpu_task(prec, 1, f"{prec} monocœur")))
            tasks.append((f"CPU {prec} multicœur", cpu_task(prec, mt, f"{prec} multicœur")))

    if cfg.do_ram:
        def ram_task(mode, label):
            def run(_progress):
                if not native.available():
                    return Measure("RAM", "Mémoire système", label, None, "Go/s", error=native.load_error)
                v = native.mem_bandwidth(mode, cfg.ram_size_mb, cfg.cpu_threads or 0, secs)
                return Measure("RAM", "Mémoire système", label, v, "Go/s",
                               details=f"tampon {cfg.ram_size_mb} Mo, multithread")
            return run
        for mode, label in (("read", "Lecture"), ("write", "Écriture"), ("copy", "Copie")):
            tasks.append((f"RAM {label.lower()}", ram_task(mode, label)))

        def lat(_progress):
            if not native.available():
                return Measure("RAM", "Mémoire système", "Latence (accès aléatoires)", None, "ns",
                               error=native.load_error)
            v = native.mem_latency(cfg.latency_size_mb, secs)
            return Measure("RAM", "Mémoire système", "Latence (accès aléatoires)", v, "ns",
                           details=f"chaînage de pointeurs, {cfg.latency_size_mb} Mo, pages 4 Ko")
        tasks.append(("RAM latence", lat))

    if cfg.do_gpu or cfg.do_vram:
        cuda_devs = cuda.devices()
        cl_devs = gpu_opencl.gpu_devices()
        cuda_names = {d.name for d in cuda_devs}
        have_cublas = cuda.cublas_available() if cuda_devs else False

        for d in cuda_devs:
            if cfg.do_gpu:
                if have_cublas:
                    for prec in cfg.gpu_precisions:
                        if prec not in cuda.GEMM_PRECISIONS:
                            continue

                        def gemm(progress, d=d, prec=prec):
                            if not cuda.gemm_supported(d, prec):
                                return Measure("GPU", d.name, GEMM_LABELS[prec], None, "TFLOPS",
                                               error=f"non pris en charge (sm_{d.cc[0]}{d.cc[1]})")
                            v, n = cuda.gemm_tflops(d, prec, secs, cfg.gemm_size or None, progress)
                            return Measure("GPU", d.name, GEMM_LABELS[prec], v, "TFLOPS",
                                           theoretical=cuda_theoretical(d, prec),
                                           details=f"cuBLAS, matrices {n}×{n}")
                        tasks.append((f"{d.name} {GEMM_LABELS[prec]}", gemm))
                else:
                    tasks.append((f"{d.name} cuBLAS", lambda _p, d=d: Measure(
                        "GPU", d.name, "GEMM (cuBLAS)", None, "TFLOPS",
                        error="cuBLAS absent : pip install nvidia-cublas-cu12 (crête FMA mesurée à la place)")))
                if cfg.gpu_fma_peak or not have_cublas:
                    cl = next((c for c in cl_devs if c.name == d.name), None)
                    if cl is not None:
                        def peak(progress, cl=cl, d=d):
                            v = gpu_opencl.fma_peak(cl, "FP32", secs, progress)
                            return Measure("GPU", d.name, "Crête FMA FP32", v, "TFLOPS",
                                           theoretical=cuda_fp32_peak(d),
                                           details="noyau OpenCL, FMA indépendantes")
                        tasks.append((f"{d.name} crête FP32", peak))
            if cfg.do_vram:
                def vram(_p, d=d):
                    v = cuda.vram_bandwidth(d, secs)
                    return Measure("VRAM", d.name, "Bande passante (copie)", v, "Go/s",
                                   theoretical=cuda.theoretical_vram_gbs(d),
                                   details=f"cuMemcpyDtoD, lecture + écriture, bus {d.bus_width} bits")
                tasks.append((f"{d.name} VRAM", vram))

        for g in cl_devs:  # GPU AMD / Intel (ou NVIDIA sans pilote CUDA)
            if g.name in cuda_names:
                continue
            if cfg.do_gpu:
                for prec in ("FP32", "FP16", "FP64"):
                    def clpeak(progress, g=g, prec=prec):
                        if not gpu_opencl.supported(g, prec):
                            return Measure("GPU", g.name, f"Crête FMA {prec}", None, "TFLOPS",
                                           error=f"{prec} non exposé par le pilote OpenCL")
                        v = gpu_opencl.fma_peak(g, prec, secs, progress)
                        return Measure("GPU", g.name, f"Crête FMA {prec}", v, "TFLOPS",
                                       theoretical=gpu_opencl.theoretical_fp32(g) if prec == "FP32" else None,
                                       details="noyau OpenCL (GEMM cuBLAS réservé à NVIDIA)")
                    tasks.append((f"{g.name} {prec}", clpeak))
            if cfg.do_vram:
                tasks.append((f"{g.name} VRAM", lambda _p, g=g: Measure(
                    "VRAM", g.name, "Bande passante (copie)", gpu_opencl.vram_bandwidth(g, secs), "Go/s",
                    details="clEnqueueCopyBuffer, lecture + écriture")))

        if cfg.do_gpu and not cuda_devs and not cl_devs:
            tasks.append(("GPU", lambda _p: Measure("GPU", "Aucun GPU détecté", "—", None, "TFLOPS",
                                                    error="Installez ou mettez à jour le pilote graphique")))

    if cfg.do_disk and native.available():
        tasks.extend(_disk_tasks(cfg, secs, drives if drives is not None else hwdetails.drives()))
    return tasks


def run_analysis(cfg: Settings, on_progress=None, on_measure=None, on_warning=None, cancel=None,
                 drives: list[dict] | None = None) -> list[Measure]:
    try:
        return _run_analysis(cfg, on_progress, on_measure, on_warning, cancel, drives)
    finally:
        _cleanup()  # aucun fichier de test ne reste sur les disques, même en cas d'arrêt


def _run_analysis(cfg, on_progress, on_measure, on_warning, cancel, drives) -> list[Measure]:
    if (cfg.do_gpu or cfg.do_vram) and on_warning:
        w = _gpu_busy_warning()
        if w:
            on_warning(w)
    tasks = build_tasks(cfg, drives)
    out = []
    for i, (label, fn) in enumerate(tasks):
        if cancel and cancel():
            raise Cancelled
        if on_progress:
            on_progress(i / len(tasks), label)

        def sub(f, i=i, label=label):
            if on_progress:
                on_progress((i + max(0.0, min(1.0, f))) / len(tasks), label)
            if cancel and cancel():
                raise InterruptedError("annulé")

        try:
            m = fn(sub)
        except InterruptedError:
            raise Cancelled
        except (cuda.CudaError, opencl.OpenCLError, native.NativeUnavailable, OSError, RuntimeError) as e:
            if ": · " in label:  # tâche disque : « C: · modèle (type) étape »
                dev, _, step_name = label.rpartition(" ")
                head = label.split(" écriture")[0].split(" lecture")[0]
                m = Measure("Disque", head, label[len(head):].strip().capitalize(), None, "Mo/s",
                            error=str(e).splitlines()[0][:300])
            else:
                cat = "CPU" if label.startswith("CPU") else "RAM" if label.startswith("RAM") else \
                    "VRAM" if label.endswith("VRAM") else "GPU"
                m = Measure(cat, label, label, None, "", error=str(e).splitlines()[0][:300])
        if cancel and cancel():
            raise Cancelled
        out.append(m)
        if on_measure:
            on_measure(m)
    if on_progress:
        on_progress(1.0, "Analyse terminée")
    return out


# --------------------------------------------------------------------------- stabilité

STABILITY_TARGETS = ["GPU FP16", "GPU FP32", "CPU FP32"]


def run_stability(target: str, minutes: float, on_sample=None, cancel=None, gemm_size: int = 0):
    """Charge prolongée. Renvoie (échantillons, résumé). Chaque échantillon ≈ 1 s."""
    monitor = Monitor()
    duration = minutes * 60
    samples = []
    is_gpu = target.startswith("GPU")
    prec = target.split()[-1]
    unit = "TFLOPS" if is_gpu else "GFLOPS"
    device_name = ""
    nv_index = None
    gen = None
    cl_dev = None

    if is_gpu:
        devs = cuda.devices()
        if devs and cuda.cublas_available():
            d = devs[0]
            device_name = d.name
            n = gemm_size or (8192 if prec != "FP64" else 4096)
            while 3 * n * n * cuda.GEMM_PRECISIONS[prec][1] > d.memory * 0.4 and n > 1024:
                n //= 2
            gen = cuda.gemm_stream(d, prec, n)
            mode = f"GEMM cuBLAS {prec} {n}×{n}"
        else:
            cls = gpu_opencl.gpu_devices()
            if not cls:
                raise RuntimeError("Aucun GPU utilisable pour le test de stabilité")
            cl_dev = cls[0]
            device_name = cl_dev.name
            if prec == "FP16" and not cl_dev.supports_fp16:
                prec = "FP32"
            mode = f"crête FMA OpenCL {prec}"
        nv_index = monitor.nvml.index_for(device_name)
    else:
        from .sysinfo import cpu_name
        device_name = cpu_name()
        mode = "FMA AVX2 multicœur"
        if not native.available():
            raise RuntimeError(native.load_error)

    start = time.perf_counter()
    interrupted = False
    try:
        while time.perf_counter() - start < duration:
            if cancel and cancel():
                interrupted = True
                break
            if gen is not None:
                vals, t0 = [], time.perf_counter()
                while time.perf_counter() - t0 < 1.0:
                    vals.append(next(gen))
                perf = statistics.fmean(vals)
            elif cl_dev is not None:
                perf = gpu_opencl.fma_peak(cl_dev, prec, 0.8)
            else:
                try:
                    perf = native.cpu_gflops(prec, 0, 1.0)
                except InterruptedError:
                    interrupted = True
                    break
            mon = monitor.sample()
            s = {"t": time.perf_counter() - start, "perf": perf}
            if is_gpu and nv_index is not None and mon["gpus"][nv_index]:
                g = mon["gpus"][nv_index]
                s.update(temp_c=g["temp_c"], clock_mhz=g["clock_mhz"], power_w=g["power_w"], throttle=g["throttle"])
            elif not is_gpu and mon["cpu"]:
                s.update(temp_c=None, clock_mhz=mon["cpu"]["mhz"], power_w=None, throttle=[])
            samples.append(s)
            if on_sample:
                on_sample(s, min(1.0, s["t"] / duration))
    finally:
        if gen is not None:
            gen.close()

    return samples, summarize(samples, target, device_name, mode, unit, minutes, interrupted)


def summarize(samples, target, device, mode, unit, minutes, interrupted=False) -> dict:
    if not samples:
        return {"target": target, "device": device, "verdict": "Aucune mesure", "unit": unit}
    perfs = [s["perf"] for s in samples]
    k = max(3, len(perfs) // 10)
    begin = statistics.fmean(perfs[:k])
    end = statistics.fmean(perfs[-k:])
    drop = (begin - end) / begin * 100 if begin else 0.0
    mean = statistics.fmean(perfs)
    cv = statistics.pstdev(perfs) / mean * 100 if mean else 0.0
    temps = [s["temp_c"] for s in samples if s.get("temp_c") is not None]
    clocks = [s["clock_mhz"] for s in samples if s.get("clock_mhz")]
    throttled = [s for s in samples if s.get("throttle")]
    reasons = sorted({r for s in throttled for r in s["throttle"]})
    if drop < 3 and cv < 5:
        verdict = "Stable"
    elif drop < 10:
        verdict = "Légère baisse sous charge"
    else:
        verdict = "Baisse importante (bridage thermique ou alimentation)"
    return {
        "target": target, "device": device, "mode": mode, "unit": unit,
        "duration_s": samples[-1]["t"], "planned_min": minutes, "interrupted": interrupted,
        "begin": begin, "end": end, "drop_pct": drop, "mean": mean, "min": min(perfs), "max": max(perfs),
        "cv_pct": cv,
        "temp_max": max(temps) if temps else None,
        "clock_begin": statistics.fmean(clocks[:k]) if clocks else None,
        "clock_end": statistics.fmean(clocks[-k:]) if clocks else None,
        "throttle_pct": len(throttled) / len(samples) * 100,
        "throttle_reasons": reasons,
        "verdict": verdict,
    }
