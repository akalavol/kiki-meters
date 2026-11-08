"""Structures de données partagées."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Measure:
    category: str             # CPU, GPU, RAM, VRAM
    device: str               # nom du matériel
    test: str                 # ex. « FP32 multicœur », « GEMM FP16 »
    value: float | None
    unit: str                 # GFLOPS, TFLOPS, Go/s, ns
    theoretical: float | None = None
    details: str = ""
    error: str = ""

    @property
    def efficiency(self):
        if self.value and self.theoretical:
            return self.value / self.theoretical
        return None

    def to_dict(self):
        return asdict(self)


@dataclass
class Settings:
    preset: str = "Standard"                 # Rapide / Standard / Précis
    do_cpu: bool = True
    do_gpu: bool = True
    do_ram: bool = True
    do_vram: bool = True
    do_disk: bool = True
    disk_letters: list = field(default_factory=list)   # vide = disques internes détectés
    disk_size_mb: int = 0                    # 0 = selon la durée (512 / 1024 / 4096 Mo)
    cpu_threads: int = 0                     # 0 = tous les threads logiques
    ram_size_mb: int = 1024
    latency_size_mb: int = 256
    gemm_size: int = 0                       # 0 = automatique
    gpu_precisions: list = field(default_factory=lambda: ["FP32", "TF32", "FP16", "BF16", "FP64"])
    gpu_fma_peak: bool = True                # crête FMA FP32 (comparaison avec la fiche technique)
    stability_target: str = "GPU FP16"
    stability_minutes: int = 5
    monitor_interval_ms: int = 1000
    ai_ctx: int = 8192                       # contexte visé pour les estimations IA

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        """Valeurs inconnues ou de mauvais type ignorées : un fichier abîmé ne bloque pas le démarrage."""
        s = cls()
        for k, v in (d or {}).items():
            if not hasattr(s, k):
                continue
            default = getattr(s, k)
            try:
                if isinstance(default, bool):
                    v = bool(v)
                elif isinstance(default, int):
                    v = int(v)
                elif isinstance(default, list):
                    v = [x for x in v if isinstance(x, str)]
                elif isinstance(default, str):
                    v = str(v)
            except (TypeError, ValueError):
                continue
            setattr(s, k, v)
        if s.preset not in PRESET_SECONDS:
            s.preset = "Standard"
        s.stability_minutes = min(max(s.stability_minutes, 1), 180)
        return s

    def to_dict(self):
        return asdict(self)


PRESET_SECONDS = {"Rapide": 0.7, "Standard": 2.0, "Précis": 5.0}
PRESET_DISK_MB = {"Rapide": 512, "Standard": 1024, "Précis": 4096}
