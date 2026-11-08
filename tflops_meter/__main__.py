"""Point d'entrée : `python -m tflops_meter` (interface) ou `python -m tflops_meter --cli` (console)."""
from __future__ import annotations

import sys


def cli(argv):
    import argparse

    from . import runner, sysinfo
    from .history import History
    from .model import PRESET_SECONDS, Settings

    p = argparse.ArgumentParser(prog="tflops_meter", description="Mesure CPU / GPU / RAM / VRAM en console")
    p.add_argument("--cli", action="store_true")
    p.add_argument("--preset", choices=list(PRESET_SECONDS), default="Standard")
    p.add_argument("--only", nargs="+", choices=["cpu", "gpu", "ram", "vram", "disk"])
    p.add_argument("--no-save", action="store_true", help="ne pas enregistrer dans l'historique")
    a = p.parse_args(argv)
    cfg = Settings(preset=a.preset)
    if a.only:
        cfg.do_cpu, cfg.do_gpu, cfg.do_ram, cfg.do_vram, cfg.do_disk = (
            k in a.only for k in ("cpu", "gpu", "ram", "vram", "disk"))
    info = sysinfo.collect()
    print(f"CPU : {info['cpu']}")
    for g in info["gpus"]:
        print(f"GPU : {g['name']} ({g['api']})")

    def show(m):
        val = f"ÉCHEC — {m.error}" if m.error else f"{m.value:,.2f} {m.unit}"
        theo = f"  (théorique {m.theoretical:,.2f})" if m.theoretical else ""
        print(f"  {m.category:5} {m.test:28} {val}{theo}")

    measures = runner.run_analysis(cfg, on_measure=show, on_warning=lambda w: print("⚠", w))
    if not a.no_save:
        History().save_run("analyse", cfg.preset, info, measures, summary={"highlights": "console"})


def main():
    if "--cli" in sys.argv:
        if sys.stdout is None:  # exécutable fenêtré : pas de console, on écrit dans un fichier
            import os

            from .paths import data_dir
            path = os.path.join(data_dir(), "resultats_console.txt")
            sys.stdout = sys.stderr = open(path, "w", encoding="utf-8", buffering=1)
        cli(sys.argv[1:])
        return
    from PySide6.QtWidgets import QApplication

    from .ui import theme
    from .ui.main_window import MainWindow

    if sys.platform == "win32":  # icône propre dans la barre des tâches
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("TflopsMeter.App")
    app = QApplication(sys.argv)
    app.setApplicationName("TFLOPS Meter")
    theme.apply(app)
    app.setWindowIcon(theme.app_icon())
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
