"""Threads de travail : les mesures ne bloquent jamais l'interface."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal

from .. import native, runner, sysinfo
from ..model import Settings


class SysInfoWorker(QThread):
    done = Signal(dict)

    def run(self):
        try:
            info = sysinfo.collect()
        except Exception as e:  # ne jamais bloquer le démarrage
            self.done.emit({"error": str(e)})
            return
        try:
            from .. import hwdetails
            info["details"] = hwdetails.collect()
        except Exception:
            info["details"] = {}
        self.done.emit(info)


class AnalysisWorker(QThread):
    progress = Signal(float, str)
    measure = Signal(object)
    warning = Signal(str)
    finished_ok = Signal(list, bool)   # mesures, annulé ?
    failed = Signal(str)

    def __init__(self, cfg: Settings, drives: list | None = None):
        super().__init__()
        self.cfg = cfg
        self.drives = drives
        self._cancel = False
        self.results = []

    def stop(self):
        self._cancel = True
        native.cancel()

    def _on_measure(self, m):
        self.results.append(m)
        self.measure.emit(m)

    def run(self):
        try:
            runner.run_analysis(self.cfg, on_progress=self.progress.emit, on_measure=self._on_measure,
                                on_warning=self.warning.emit, cancel=lambda: self._cancel, drives=self.drives)
            self.finished_ok.emit(self.results, False)
        except runner.Cancelled:
            self.finished_ok.emit(self.results, True)
        except Exception:
            self.failed.emit(traceback.format_exc(limit=3))


class StabilityWorker(QThread):
    sample = Signal(dict, float)
    finished_ok = Signal(list, dict)
    failed = Signal(str)

    def __init__(self, target: str, minutes: float, gemm_size: int = 0):
        super().__init__()
        self.target, self.minutes, self.gemm_size = target, minutes, gemm_size
        self._cancel = False

    def stop(self):
        self._cancel = True
        native.cancel()

    def run(self):
        try:
            samples, summary = runner.run_stability(self.target, self.minutes, on_sample=self.sample.emit,
                                                    cancel=lambda: self._cancel, gemm_size=self.gemm_size)
            self.finished_ok.emit(samples, summary)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__} : {e}")
