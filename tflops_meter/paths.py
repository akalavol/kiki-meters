"""Emplacements des ressources (fonctionne aussi dans l'exécutable PyInstaller)."""
from __future__ import annotations

import os
import sys


def resource_dir() -> str:
    """Dossier du paquet tflops_meter (ou son équivalent dans le bundle PyInstaller)."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return os.path.join(base, "tflops_meter")
    return os.path.dirname(os.path.abspath(__file__))


def data_dir() -> str:
    """Dossier utilisateur pour l'historique et les réglages."""
    root = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    path = os.path.join(root, "TflopsMeter")
    os.makedirs(path, exist_ok=True)
    return path
