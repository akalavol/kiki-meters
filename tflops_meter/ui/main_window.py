"""Fenêtre principale de TFLOPS Meter."""
from __future__ import annotations

import json
import os
from datetime import datetime

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
                               QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
                               QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import __version__
from ..history import History, export_csv, export_json
from ..model import PRESET_SECONDS, Measure, Settings
from ..monitor import Monitor
from ..paths import data_dir
from ..runner import STABILITY_TARGETS
from . import theme
from .ai_tab import AiTab
from .catalog_tab import CatalogTab
from .image_tab import ImageCatalogTab
from .charts import BarChart, LiveChart
from .workers import AnalysisWorker, StabilityWorker, SysInfoWorker

SETTINGS_FILE = os.path.join(data_dir(), "parametres.json")
DECIMALS = {"TFLOPS": 2, "GFLOPS": 1, "Go/s": 1, "ns": 1, "%": 1, "°C": 0, "MHz": 0, "W": 0, "tok/s": 1,
            "Mo/s": 0}
KIND_LABELS = {"analyse": "Analyse", "stabilite": "Stabilité", "ia": "IA locale"}


def fmt(v, unit: str = "", dec: int | None = None) -> str:
    if v is None:
        return "—"
    d = DECIMALS.get(unit, 2) if dec is None else dec
    s = f"{v:,.{d}f}".replace(",", " ").replace(".", ",")
    return f"{s} {unit}".strip()


def signed_pct(v: float) -> str:
    v = 0.0 if abs(v) < 0.05 else v  # évite « -0,0 % »
    return f"{v:+.1f} %".replace(".", ",").replace("+0,0", "0,0")


def eff_color(e):
    if e is None:
        return theme.MUTED
    return theme.GOOD if e >= 0.75 else theme.WARN if e >= 0.5 else theme.BAD


class Card(QFrame):
    def __init__(self, title: str, color: str):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        t = QLabel(title.upper())
        t.setObjectName("cardTitle")
        self.value = QLabel("—")
        self.value.setObjectName("cardValue")
        self.value.setStyleSheet(f"color: {color};")
        self.sub = QLabel(" ")
        self.sub.setObjectName("cardSub")
        self.sub.setWordWrap(True)
        self.sub.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.sub.setFixedHeight(self.sub.fontMetrics().lineSpacing() * 2 + 4)  # cartes alignées
        lay.addWidget(t)
        lay.addWidget(self.value)
        lay.addWidget(self.sub)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set(self, value: str, sub: str = " "):
        self.value.setText(value)
        self.sub.setText(sub)


def results_tree() -> QTreeWidget:
    tree = QTreeWidget()
    tree.setColumnCount(5)
    tree.setHeaderLabels(["Test", "Résultat", "Théorique", "Efficacité", "Détails"])
    tree.setAlternatingRowColors(True)
    tree.setRootIsDecorated(True)
    tree.setUniformRowHeights(True)
    h = tree.header()
    h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
    for i in (1, 2, 3):
        h.setSectionResizeMode(i, QHeaderView.ResizeToContents)
    h.setSectionResizeMode(4, QHeaderView.Stretch)
    return tree


def add_measure(tree: QTreeWidget, groups: dict, m: dict):
    """Ajoute une mesure (dict au format Measure) sous son groupe catégorie/matériel."""
    key = (m["category"], m["device"])
    parent = groups.get(key)
    if parent is None:
        parent = QTreeWidgetItem([f"{m['category']}  ·  {m['device']}"])
        f = parent.font(0)
        f.setBold(True)
        parent.setFont(0, f)
        parent.setForeground(0, QColor(theme.CATEGORY_COLORS.get(m["category"], theme.TEXT)))
        tree.addTopLevelItem(parent)
        parent.setExpanded(True)
        groups[key] = parent
    v, t = m.get("value"), m.get("theoretical")
    eff = v / t if v and t else None
    item = QTreeWidgetItem([
        m["test"],
        fmt(v, m["unit"]) if not m.get("error") else "échec",
        fmt(t, m["unit"]) if t else "",
        f"{eff * 100:.0f} %" if eff else "",
        m.get("error") or m.get("details") or "",
    ])
    for col in (1, 2, 3):
        item.setTextAlignment(col, Qt.AlignRight | Qt.AlignVCenter)
    f = item.font(1)
    f.setBold(True)
    item.setFont(1, f)
    if m.get("error"):
        item.setForeground(1, QColor(theme.BAD))
        item.setForeground(4, QColor(theme.BAD))
    else:
        item.setForeground(4, QColor(theme.MUTED))
    if eff:
        item.setForeground(3, QColor(eff_color(eff)))
    item.setToolTip(4, item.text(4))
    parent.addChild(item)


# =========================================================================== Analyse

class AnalysisTab(QWidget):
    saved = Signal(int)

    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self.worker: AnalysisWorker | None = None
        self.measures: list[Measure] = []
        self.groups = {}
        self.last_run_id = None

        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        # ---- colonne gauche
        left = QVBoxLayout()
        left.setSpacing(10)
        sys_box = QGroupBox("Système")
        sl = QVBoxLayout(sys_box)
        self.sys_label = QLabel("Détection du matériel…")
        self.sys_label.setWordWrap(True)
        self.sys_label.setTextFormat(Qt.RichText)
        sl.addWidget(self.sys_label)
        left.addWidget(sys_box)

        tests = QGroupBox("Tests à lancer")
        tl = QVBoxLayout(tests)
        self.cb = {}
        for key, text, tip in (
            ("do_cpu", "Processeur (CPU)", "FP32 / FP64, monocœur et multicœur — GFLOPS"),
            ("do_gpu", "Carte graphique (GPU)", "GEMM FP32, TF32, FP16, BF16, FP64 — TFLOPS"),
            ("do_ram", "Mémoire vive (RAM)", "Lecture, écriture, copie (Go/s) et latence (ns)"),
            ("do_vram", "Mémoire vidéo (VRAM)", "Bande passante GPU — Go/s"),
            ("do_disk", "Stockage (disques)", "Séquentiel et 4K aléatoire — Mo/s"),
        ):
            c = QCheckBox(text)
            c.setToolTip(tip)
            c.setChecked(getattr(main.settings, key))
            c.toggled.connect(lambda v, k=key: self._set(k, v))
            tl.addWidget(c)
            sub = QLabel(tip)
            sub.setObjectName("muted")
            sub.setStyleSheet("margin-left: 26px; font-size: 8.5pt;")
            tl.addWidget(sub)
            self.cb[key] = c
        tl.addSpacing(6)
        row = QHBoxLayout()
        row.addWidget(QLabel("Durée :"))
        self.preset = QComboBox()
        for p, s in PRESET_SECONDS.items():
            self.preset.addItem(f"{p}  (~{s:g} s par test)", p)
        self.preset.setCurrentIndex(max(0, list(PRESET_SECONDS).index(main.settings.preset)))
        self.preset.currentIndexChanged.connect(lambda _: self._set("preset", self.preset.currentData()))
        row.addWidget(self.preset, 1)
        tl.addLayout(row)
        left.addWidget(tests)

        self.start_btn = QPushButton("▶  Lancer l'analyse")
        self.start_btn.setObjectName("primary")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("■  Arrêter")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(self.stop)
        left.addWidget(self.start_btn)
        left.addWidget(self.stop_btn)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        left.addWidget(self.progress)
        self.status = QLabel("Prêt.")
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        self.warning = QLabel()
        self.warning.setObjectName("warning")
        self.warning.setWordWrap(True)
        self.warning.setVisible(False)
        left.addWidget(self.warning)
        left.addStretch(1)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(330)
        root.addWidget(left_w)

        # ---- zone de résultats
        right = QVBoxLayout()
        right.setSpacing(12)
        cards = QHBoxLayout()
        cards.setSpacing(12)
        self.card_gpu = Card("GPU — meilleur résultat", theme.CATEGORY_COLORS["GPU"])
        self.card_cpu = Card("CPU — FP32 multicœur", theme.CATEGORY_COLORS["CPU"])
        self.card_ram = Card("RAM — lecture", theme.CATEGORY_COLORS["RAM"])
        self.card_vram = Card("VRAM — bande passante", theme.CATEGORY_COLORS["VRAM"])
        self.card_disk = Card("Disque — lecture séq.", theme.CATEGORY_COLORS["Disque"])
        for c in (self.card_gpu, self.card_cpu, self.card_ram, self.card_vram, self.card_disk):
            cards.addWidget(c)
        right.addLayout(cards)

        split = QSplitter(Qt.Vertical)
        self.tree = results_tree()
        split.addWidget(self.tree)
        charts = QWidget()
        cl = QHBoxLayout(charts)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(12)
        self.chart_gpu = BarChart("GPU", "TFLOPS", theme.CATEGORY_COLORS["GPU"])
        self.chart_cpu = BarChart("CPU", "GFLOPS", theme.CATEGORY_COLORS["CPU"])
        cl.addWidget(self.chart_gpu, 3)
        cl.addWidget(self.chart_cpu, 2)
        split.addWidget(charts)
        split.setSizes([330, 300])
        right.addWidget(split, 1)

        exp = QHBoxLayout()
        img_btn = QPushButton("🖼  Image pour Discord (JPG)")
        img_btn.clicked.connect(main.save_share_image)
        exp.addWidget(img_btn)
        img_copy = QPushButton("Copier l'image")
        img_copy.setToolTip("Copie l'image dans le presse-papiers : collez-la directement dans Discord (Ctrl+V)")
        img_copy.clicked.connect(main.copy_share_image)
        exp.addWidget(img_copy)
        exp.addStretch(1)
        self.copy_btn = QPushButton("Copier le résumé")
        self.copy_btn.clicked.connect(self.copy_summary)
        self.json_btn = QPushButton("Exporter JSON")
        self.json_btn.clicked.connect(lambda: self.export("json"))
        self.csv_btn = QPushButton("Exporter CSV")
        self.csv_btn.clicked.connect(lambda: self.export("csv"))
        for b in (self.copy_btn, self.json_btn, self.csv_btn):
            b.setEnabled(False)
            exp.addWidget(b)
        right.addLayout(exp)
        root.addLayout(right, 1)

    def _set(self, key, value):
        setattr(self.main.settings, key, value)
        self.main.save_settings()

    def show_sysinfo(self, info: dict):
        if "error" in info:
            self.sys_label.setText(f"<span style='color:{theme.BAD}'>{info['error']}</span>")
            return
        m = theme.MUTED
        det = info.get("details") or {}
        ram = det.get("ram")
        ram_txt = (f"RAM {fmt(ram['total_gb'], 'Go', 0)} {ram['type']} · {ram['layout']} · {ram['speed_mts']} MT/s"
                   if ram else f"RAM {fmt(info.get('ram_gb'), 'Go', 0)}")
        lines = [f"<b>{info['cpu']}</b>",
                 f"<span style='color:{m}'>{info.get('cores') or '?'} cœurs · {info['threads']} threads · "
                 f"{' / '.join(info.get('cpu_features') or []) or 'SSE2'}</span>",
                 f"<span style='color:{m}'>{ram_txt}</span>"]
        if det.get("board"):
            lines.append(f"<span style='color:{m}'>Carte mère {det['board']}</span>")
        for g in info.get("gpus", []):
            cc = f" · sm_{g['compute_capability'].replace('.', '')}" if g.get("compute_capability") else ""
            lines.append(f"<br><b>{g['name']}</b>")
            lines.append(f"<span style='color:{m}'>{fmt(g['memory_gb'], 'Go', 0)} · {g['units']} unités · "
                         f"{g['clock_mhz']} MHz · {g['api']}{cc}</span>")
        if not info.get("gpus"):
            lines.append(f"<br><span style='color:{theme.WARN}'>Aucun GPU détecté</span>")
        disks = [d for d in det.get("disks") or [] if d.get("kind") != "USB"]
        if disks:
            lines.append("<br>" + "<br>".join(
                f"<span style='color:{m}'>{', '.join(x + ':' for x in d['letters']) or '—'} {d['model']} · "
                f"{fmt(d['size_gb'] / 1000, 'To', 1) if d['size_gb'] >= 1000 else fmt(d['size_gb'], 'Go', 0)} · "
                f"{d['kind']}</span>" for d in disks))
        libs = []
        libs.append(("Module natif", info.get("native")))
        libs.append(("cuBLAS", info.get("cublas")))
        lines.append("<br>" + " · ".join(
            f"<span style='color:{theme.GOOD if ok else theme.BAD}'>{'✔' if ok else '✖'} {n}</span>"
            for n, ok in libs))
        self.sys_label.setText("<br>".join(lines))
        self.sys_label.setToolTip(info.get("cublas_info") or "")

    # ---- exécution
    def start(self):
        if not self.main.can_start():
            return
        cfg = Settings.from_dict(self.main.settings.to_dict())
        if not (cfg.do_cpu or cfg.do_gpu or cfg.do_ram or cfg.do_vram or cfg.do_disk):
            QMessageBox.information(self, "TFLOPS Meter", "Cochez au moins un test.")
            return
        self.tree.clear()
        self.groups.clear()
        self.measures = []
        self.last_run_id = None
        for c in (self.card_gpu, self.card_cpu, self.card_ram, self.card_vram, self.card_disk):
            c.set("—")
        self.chart_gpu.set_data([])
        self.chart_cpu.set_data([])
        self.warning.setVisible(False)
        for b in (self.copy_btn, self.json_btn, self.csv_btn):
            b.setEnabled(False)
        self.worker = AnalysisWorker(cfg, (self.main.sysinfo.get("details") or {}).get("drives"))
        self.worker.progress.connect(self.on_progress)
        self.worker.measure.connect(self.on_measure)
        self.worker.warning.connect(self.on_warning)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.main.set_busy(True)
        self.start_btn.setVisible(False)
        self.stop_btn.setVisible(True)
        self.stop_btn.setEnabled(True)
        self.status.setText("Préparation…")
        self.worker.start()

    def stop(self):
        if self.worker:
            self.stop_btn.setEnabled(False)
            self.status.setText("Arrêt en cours…")
            self.worker.stop()

    def on_progress(self, f, label):
        self.progress.setValue(int(f * 1000))
        self.status.setText(label if f >= 1 else f"En cours : {label}")

    def on_warning(self, text):
        self.warning.setText("⚠  " + text)
        self.warning.setVisible(True)

    def on_measure(self, m: Measure):
        self.measures.append(m)
        add_measure(self.tree, self.groups, m.to_dict())
        self.tree.scrollToBottom()
        self.refresh_highlights()

    def refresh_highlights(self):
        ok = [m for m in self.measures if m.value is not None and not m.error]
        gpu = [m for m in ok if m.category == "GPU"]
        if gpu:
            best = max(gpu, key=lambda m: m.value)
            self.card_gpu.set(fmt(best.value, "TFLOPS"), f"{best.test} · {best.device}")
        cpu = next((m for m in ok if m.category == "CPU" and m.test == "FP32 multicœur"), None)
        if cpu:
            mono = next((m for m in ok if m.category == "CPU" and m.test == "FP32 monocœur"), None)
            self.card_cpu.set(fmt(cpu.value, "GFLOPS"),
                              f"monocœur {fmt(mono.value, 'GFLOPS')}" if mono else cpu.device)
        ram = next((m for m in ok if m.category == "RAM" and m.test == "Lecture"), None)
        lat = next((m for m in ok if m.category == "RAM" and m.unit == "ns"), None)
        if ram or lat:
            self.card_ram.set(fmt(ram.value, "Go/s") if ram else "—",
                              f"latence {fmt(lat.value, 'ns')}" if lat else " ")
        vram = [m for m in ok if m.category == "VRAM"]
        if vram:
            v = vram[0]
            self.card_vram.set(fmt(v.value, "Go/s"),
                               f"théorique {fmt(v.theoretical, 'Go/s')}" if v.theoretical else v.device)
        disk = [m for m in ok if m.category == "Disque" and m.test == "Lecture séquentielle"]
        if disk:
            b = max(disk, key=lambda m: m.value)
            w = next((m for m in ok if m.device == b.device and m.test == "Écriture séquentielle"), None)
            self.card_disk.set(fmt(b.value, "Mo/s"), (f"écriture {fmt(w.value, 'Mo/s')} · " if w else "")
                               + b.device.split(" · ")[0])
        multi_gpu = len({m.device for m in gpu}) > 1
        self.chart_gpu.set_data([((f"{m.test} · {m.device}" if multi_gpu else m.test), m.value, m.theoretical)
                                 for m in gpu])
        self.chart_cpu.set_data([(m.test, m.value, None) for m in ok if m.category == "CPU"])

    def highlights_text(self) -> str:
        ok = [m for m in self.measures if m.value is not None and not m.error]
        parts = []
        gpu = [m for m in ok if m.category == "GPU"]
        if gpu:
            b = max(gpu, key=lambda m: m.value)
            parts.append(f"GPU {fmt(b.value, 'TFLOPS')} ({b.test.replace('GEMM ', '')})")
        cpu = next((m for m in ok if m.test == "FP32 multicœur"), None)
        if cpu:
            parts.append(f"CPU {fmt(cpu.value, 'GFLOPS')}")
        ram = next((m for m in ok if m.category == "RAM" and m.test == "Lecture"), None)
        if ram:
            parts.append(f"RAM {fmt(ram.value, 'Go/s')}")
        vram = next((m for m in ok if m.category == "VRAM"), None)
        if vram:
            parts.append(f"VRAM {fmt(vram.value, 'Go/s')}")
        disk = [m for m in ok if m.category == "Disque" and m.test == "Lecture séquentielle"]
        if disk:
            parts.append(f"Disque {fmt(max(m.value for m in disk), 'Mo/s')}")
        return " · ".join(parts)

    def on_finished(self, results, cancelled):
        self.main.set_busy(False)
        self.start_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.progress.setValue(0 if cancelled else 1000)
        if results:
            self.last_run_id = self.main.history.save_run(
                "analyse", self.main.settings.preset, self.main.sysinfo, results,
                summary={"highlights": self.highlights_text(), "interrupted": cancelled},
                label="Interrompue" if cancelled else "")
            self.saved.emit(self.last_run_id)
            for b in (self.copy_btn, self.json_btn, self.csv_btn):
                b.setEnabled(True)
            self.main.ai_tab.recompute()  # les estimations IA utilisent les nouveaux débits
        failed = sum(1 for m in results if m.error)
        msg = "Analyse arrêtée." if cancelled else "Analyse terminée et enregistrée dans l'historique."
        if failed:
            msg += f" {failed} test(s) en échec (voir la colonne Détails)."
        self.status.setText(msg)

    def on_failed(self, text):
        self.main.set_busy(False)
        self.start_btn.setVisible(True)
        self.stop_btn.setVisible(False)
        self.status.setText("Erreur inattendue.")
        QMessageBox.critical(self, "TFLOPS Meter", text)

    def copy_summary(self):
        lines = [f"TFLOPS Meter — {datetime.now():%d/%m/%Y %H:%M}"]
        for m in self.measures:
            val = "échec : " + m.error if m.error else fmt(m.value, m.unit)
            lines.append(f"{m.category:5} {m.device} — {m.test} : {val}")
        QGuiApplication.clipboard().setText("\n".join(lines))
        self.status.setText("Résumé copié dans le presse-papiers.")

    def export(self, kind):
        if self.last_run_id is None:
            return
        self.main.export_runs([self.last_run_id], kind)


# =========================================================================== Stabilité

class StabilityTab(QWidget):
    saved = Signal(int)

    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self.worker: StabilityWorker | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Charge :"))
        self.target = QComboBox()
        for t in STABILITY_TARGETS:
            self.target.addItem({"GPU FP16": "GPU — GEMM FP16 (Tensor Cores)", "GPU FP32": "GPU — GEMM FP32",
                                 "CPU FP32": "CPU — FMA FP32 tous les cœurs"}[t], t)
        idx = self.target.findData(main.settings.stability_target)
        self.target.setCurrentIndex(max(0, idx))
        bar.addWidget(self.target)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Durée :"))
        self.minutes = QSpinBox()
        self.minutes.setRange(1, 180)
        self.minutes.setSuffix(" min")
        self.minutes.setValue(main.settings.stability_minutes)
        bar.addWidget(self.minutes)
        bar.addSpacing(12)
        self.start_btn = QPushButton("▶  Démarrer")
        self.start_btn.setObjectName("primary")
        self.start_btn.setStyleSheet("padding: 8px 22px; font-size: 10.5pt;")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("■  Arrêter")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        bar.addWidget(self.start_btn)
        bar.addWidget(self.stop_btn)
        bar.addSpacing(12)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        bar.addWidget(self.progress, 1)
        self.elapsed = QLabel("")
        self.elapsed.setObjectName("muted")
        self.elapsed.setMinimumWidth(110)
        bar.addWidget(self.elapsed)
        root.addLayout(bar)

        hint = QLabel("Fait tourner une charge continue et suit l'évolution des performances, de la température, "
                      "des fréquences et de la puissance. Une baisse au fil du temps révèle un bridage "
                      "thermique ou électrique.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        root.addWidget(hint)

        body = QHBoxLayout()
        charts = QVBoxLayout()
        self.chart_perf = LiveChart("Performance et température", ("TFLOPS", theme.GOOD), ("°C", theme.BAD))
        self.chart_clock = LiveChart("Fréquence et puissance", ("MHz", theme.ACCENT), ("W", theme.WARN))
        charts.addWidget(self.chart_perf, 1)
        charts.addWidget(self.chart_clock, 1)
        body.addLayout(charts, 1)

        summ = QGroupBox("Bilan")
        summ.setFixedWidth(300)
        sl = QVBoxLayout(summ)
        self.verdict = QLabel("—")
        self.verdict.setObjectName("verdict")
        self.verdict.setWordWrap(True)
        sl.addWidget(self.verdict)
        grid = QGridLayout()
        grid.setVerticalSpacing(6)
        self.vals = {}
        for i, (k, label) in enumerate((
            ("now", "Actuel"), ("begin", "Début"), ("end", "Fin"), ("drop", "Variation"), ("mean", "Moyenne"),
            ("minmax", "Min / max"), ("cv", "Fluctuation"), ("temp", "Température max"),
            ("clock", "Fréquence début → fin"), ("throttle", "Temps bridé"), ("reasons", "Causes"),
        )):
            lab = QLabel(label)
            lab.setObjectName("muted")
            val = QLabel("—")
            val.setWordWrap(True)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(lab, i, 0)
            grid.addWidget(val, i, 1)
            self.vals[k] = val
        sl.addLayout(grid)
        sl.addStretch(1)
        body.addWidget(summ)
        root.addLayout(body, 1)

    def start(self):
        if not self.main.can_start():
            return
        target = self.target.currentData()
        self.main.settings.stability_target = target
        self.main.settings.stability_minutes = self.minutes.value()
        self.main.save_settings()
        unit = "TFLOPS" if target.startswith("GPU") else "GFLOPS"
        self.unit = unit
        self.chart_perf.set_titles(unit, "°C")
        self.chart_perf.clear()
        self.chart_clock.clear()
        for v in self.vals.values():
            v.setText("—")
        self.verdict.setText("Mesure en cours…")
        self.verdict.setStyleSheet(f"color: {theme.TEXT};")
        self.worker = StabilityWorker(target, self.minutes.value(), self.main.settings.gemm_size)
        self.worker.sample.connect(self.on_sample)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.main.set_busy(True)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.target.setEnabled(False)
        self.minutes.setEnabled(False)
        self.worker.start()

    def stop(self):
        if self.worker:
            self.stop_btn.setEnabled(False)
            self.worker.stop()

    def on_sample(self, s: dict, f: float):
        self.progress.setValue(int(f * 1000))
        t = s["t"]
        self.elapsed.setText(f"{int(t // 60):02d}:{int(t % 60):02d}")
        self.chart_perf.add(t, s["perf"], s.get("temp_c"))
        self.chart_clock.add(t, s.get("clock_mhz"), s.get("power_w"))
        self.vals["now"].setText(fmt(s["perf"], self.unit))
        if s.get("temp_c") is not None:
            self.vals["temp"].setText(fmt(max(s["temp_c"], self._num(self.vals["temp"])), "°C"))
        if s.get("throttle"):
            self.vals["reasons"].setText(", ".join(s["throttle"]))

    @staticmethod
    def _num(label):
        try:
            return float(label.text().split()[0].replace(",", "."))
        except (ValueError, IndexError):
            return 0.0

    def _reset(self):
        self.main.set_busy(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.target.setEnabled(True)
        self.minutes.setEnabled(True)

    def show_summary(self, s: dict):
        u = s.get("unit", "")
        if "begin" not in s:
            self.verdict.setText(s.get("verdict", "—"))
            return
        color = theme.GOOD if s["verdict"] == "Stable" else theme.WARN if s["drop_pct"] < 10 else theme.BAD
        self.verdict.setText(s["verdict"] + (" (interrompu)" if s.get("interrupted") else ""))
        self.verdict.setStyleSheet(f"color: {color};")
        self.vals["begin"].setText(fmt(s["begin"], u))
        self.vals["end"].setText(fmt(s["end"], u))
        self.vals["drop"].setText(signed_pct(-s["drop_pct"]))
        self.vals["mean"].setText(fmt(s["mean"], u))
        self.vals["minmax"].setText(f"{fmt(s['min'], '', 1)} / {fmt(s['max'], u, 1)}")
        self.vals["cv"].setText(f"± {s['cv_pct']:.1f} %".replace(".", ","))
        self.vals["temp"].setText(fmt(s.get("temp_max"), "°C") if s.get("temp_max") is not None
                                  else "non disponible")
        if s.get("clock_begin"):
            self.vals["clock"].setText(f"{s['clock_begin']:.0f} → {s['clock_end']:.0f} MHz")
        self.vals["throttle"].setText(f"{s['throttle_pct']:.0f} %")
        self.vals["reasons"].setText(", ".join(s.get("throttle_reasons") or []) or "aucune")

    def on_finished(self, samples, summary):
        self._reset()
        self.show_summary(summary)
        if samples:
            u = summary.get("unit", "")
            dev = summary.get("device", "")
            measures = [
                Measure("Stabilité", dev, "Performance au début", summary.get("begin"), u),
                Measure("Stabilité", dev, "Performance à la fin", summary.get("end"), u),
                Measure("Stabilité", dev, "Variation", -summary.get("drop_pct", 0), "%",
                        details=summary.get("verdict", "")),
            ]
            if summary.get("temp_max") is not None:
                measures.append(Measure("Stabilité", dev, "Température max", summary["temp_max"], "°C"))
            run_id = self.main.history.save_run(
                "stabilite", f"{summary.get('planned_min')} min", self.main.sysinfo, measures,
                summary={**summary, "highlights": f"{summary.get('target')} · {summary.get('verdict')} · "
                                                  f"{signed_pct(-summary.get('drop_pct', 0))}"},
                samples=samples, label="Interrompu" if summary.get("interrupted") else "")
            self.saved.emit(run_id)

    def on_failed(self, text):
        self._reset()
        self.verdict.setText("Erreur")
        self.verdict.setStyleSheet(f"color: {theme.BAD};")
        QMessageBox.critical(self, "TFLOPS Meter", text)


# =========================================================================== Historique

class CompareDialog(QDialog):
    def __init__(self, parent, a: dict, b: dict):
        super().__init__(parent)
        self.setWindowTitle("Comparaison")
        self.resize(900, 560)
        lay = QVBoxLayout(self)
        head = QLabel(f"<b>A</b> : {a['created'].replace('T', ' ')} {a.get('label') or ''} &nbsp;&nbsp; "
                      f"<b>B</b> : {b['created'].replace('T', ' ')} {b.get('label') or ''}")
        lay.addWidget(head)
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(["Catégorie", "Test", "Matériel", "A", "B", "Écart B / A"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setAlternatingRowColors(True)
        idx_b = {(r["category"], r["device"], r["test"]): r for r in b["results"]}
        keys = [(r["category"], r["device"], r["test"]) for r in a["results"]]
        keys += [k for k in idx_b if k not in set(keys)]
        idx_a = {(r["category"], r["device"], r["test"]): r for r in a["results"]}
        table.setRowCount(len(keys))
        for i, k in enumerate(keys):
            ra, rb = idx_a.get(k), idx_b.get(k)
            unit = (ra or rb)["unit"]
            va = ra["value"] if ra else None
            vb = rb["value"] if rb else None
            cells = [k[0], k[2], k[1], fmt(va, unit), fmt(vb, unit), ""]
            delta = None
            if va and vb:
                delta = (vb - va) / abs(va) * 100
                if unit == "ns":
                    delta = -delta  # latence : plus bas = mieux
                cells[5] = f"{delta:+.1f} %".replace(".", ",")
            for j, c in enumerate(cells):
                it = QTableWidgetItem(c)
                if j >= 3:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if j == 5 and delta is not None:
                    it.setForeground(QColor(theme.GOOD if delta > 2 else theme.BAD if delta < -2 else theme.MUTED))
                table.setItem(i, j, it)
        table.resizeColumnsToContents()
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        lay.addWidget(table)
        note = QLabel("Écart positif = B meilleur que A (pour la latence, plus bas est meilleur).")
        note.setObjectName("muted")
        lay.addWidget(note)


class HistoryTab(QWidget):
    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        bar = QHBoxLayout()
        self.btns = {}
        for key, text in (("compare", "Comparer (2 lignes)"), ("rename", "Renommer"), ("json", "Exporter JSON"),
                          ("csv", "Exporter CSV"), ("delete", "Supprimer")):
            b = QPushButton(text)
            b.clicked.connect(getattr(self, f"do_{key}"))
            bar.addWidget(b)
            self.btns[key] = b
        self.btns["delete"].setObjectName("danger")
        bar.addStretch(1)
        refresh = QPushButton("Actualiser")
        refresh.clicked.connect(self.reload)
        bar.addWidget(refresh)
        root.addLayout(bar)

        split = QSplitter(Qt.Vertical)
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Date", "Type", "Durée", "Libellé", "Résumé"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self.on_select)
        split.addWidget(self.table)

        detail = QWidget()
        dl = QHBoxLayout(detail)
        dl.setContentsMargins(0, 0, 0, 0)
        self.tree = results_tree()
        dl.addWidget(self.tree, 3)
        self.chart = LiveChart("Évolution (test de stabilité)", ("perf", theme.GOOD), ("°C", theme.BAD))
        self.chart.setVisible(False)
        dl.addWidget(self.chart, 2)
        split.addWidget(detail)
        split.setSizes([260, 380])
        root.addWidget(split, 1)
        self.reload()

    def selected_ids(self) -> list[int]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.table.item(r, 0).data(Qt.UserRole) for r in rows]

    def reload(self):
        runs = self.main.history.runs()
        self.table.setRowCount(len(runs))
        for i, r in enumerate(runs):
            date = datetime.fromisoformat(r["created"]).strftime("%d/%m/%Y %H:%M")
            cells = [date, KIND_LABELS.get(r["kind"], r["kind"]), r.get("preset") or "",
                     r.get("label") or "", r["summary"].get("highlights", "")]
            for j, c in enumerate(cells):
                it = QTableWidgetItem(c)
                if j == 0:
                    it.setData(Qt.UserRole, r["id"])
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.on_select()

    def on_select(self):
        ids = self.selected_ids()
        self.btns["compare"].setEnabled(len(ids) == 2)
        for k in ("rename",):
            self.btns[k].setEnabled(len(ids) == 1)
        for k in ("json", "csv", "delete"):
            self.btns[k].setEnabled(bool(ids))
        self.tree.clear()
        self.chart.setVisible(False)
        if len(ids) != 1:
            return
        run = self.main.history.run(ids[0])
        if not run:
            return
        groups = {}
        for m in run["results"]:
            add_measure(self.tree, groups, m)
        if run["samples"]:
            unit = run["summary"].get("unit", "perf")
            self.chart.set_titles(unit, "°C")
            self.chart.load([(s["t"], s["perf"], s["temp_c"]) for s in run["samples"]])
            self.chart.setVisible(True)

    def do_compare(self):
        ids = self.selected_ids()
        if len(ids) == 2:
            a, b = sorted((self.main.history.run(i) for i in ids), key=lambda r: r["created"])
            CompareDialog(self, a, b).exec()

    def do_rename(self):
        ids = self.selected_ids()
        if len(ids) != 1:
            return
        text, ok = QInputDialog.getText(self, "Renommer", "Libellé :")
        if ok:
            self.main.history.rename(ids[0], text)
            self.reload()

    def do_json(self):
        self.main.export_runs(self.selected_ids(), "json")

    def do_csv(self):
        self.main.export_runs(self.selected_ids(), "csv")

    def do_delete(self):
        ids = self.selected_ids()
        if ids and QMessageBox.question(self, "Supprimer",
                                        f"Supprimer {len(ids)} mesure(s) de l'historique ?") == QMessageBox.Yes:
            for i in ids:
                self.main.history.delete(i)
            self.reload()


# =========================================================================== Paramètres

class SettingsTab(QScrollArea):
    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        s = main.settings
        self.setWidgetResizable(True)
        w = QWidget()
        self.setWidget(w)
        root = QVBoxLayout(w)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        cpu = QGroupBox("Processeur et mémoire")
        f = QFormLayout(cpu)
        f.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.threads = QSpinBox()
        self.threads.setRange(0, 512)
        self.threads.setSpecialValueText("Automatique (tous)")
        for sb in (self.threads,):
            sb.setMinimumWidth(190)
        self.threads.setValue(s.cpu_threads)
        self.threads.valueChanged.connect(lambda v: self._set("cpu_threads", v))
        f.addRow("Threads du test multicœur :", self.threads)
        self.ram = QSpinBox()
        self.ram.setRange(64, 16384)
        self.ram.setSingleStep(256)
        self.ram.setSuffix(" Mo")
        self.ram.setValue(s.ram_size_mb)
        self.ram.valueChanged.connect(lambda v: self._set("ram_size_mb", v))
        f.addRow("Tampon de débit RAM :", self.ram)
        self.lat = QSpinBox()
        self.lat.setRange(16, 4096)
        self.lat.setSingleStep(64)
        self.lat.setSuffix(" Mo")
        self.lat.setValue(s.latency_size_mb)
        self.lat.valueChanged.connect(lambda v: self._set("latency_size_mb", v))
        f.addRow("Zone du test de latence :", self.lat)
        root.addWidget(cpu)

        gpu = QGroupBox("Carte graphique")
        g = QVBoxLayout(gpu)
        g.addWidget(QLabel("Précisions GEMM mesurées (cuBLAS, cartes NVIDIA) :"))
        row = QHBoxLayout()
        row.setSpacing(22)
        self.prec = {}
        tips = {"FP32": "Simple précision classique", "TF32": "FP32 accéléré par Tensor Cores (Ampere et +)",
                "FP16": "Demi-précision sur Tensor Cores — IA", "BF16": "Brain float 16 — IA (Ampere et +)",
                "FP64": "Double précision — très bridée sur les cartes grand public"}
        for p in ("FP32", "TF32", "FP16", "BF16", "FP64"):
            c = QCheckBox(p)
            c.setToolTip(tips[p])
            c.setChecked(p in s.gpu_precisions)
            c.toggled.connect(self._precisions)
            row.addWidget(c)
            self.prec[p] = c
        row.addStretch(1)
        g.addLayout(row)
        self.fma = QCheckBox("Mesurer aussi la crête FMA FP32 (comparable à la fiche technique)")
        self.fma.setChecked(s.gpu_fma_peak)
        self.fma.toggled.connect(lambda v: self._set("gpu_fma_peak", v))
        g.addWidget(self.fma)
        fr = QFormLayout()
        fr.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.gemm = QComboBox()
        for label, val in (("Automatique", 0), ("2048", 2048), ("4096", 4096), ("8192", 8192), ("16384", 16384)):
            self.gemm.addItem(label, val)
        self.gemm.setCurrentIndex(max(0, self.gemm.findData(s.gemm_size)))
        self.gemm.currentIndexChanged.connect(lambda _: self._set("gemm_size", self.gemm.currentData()))
        fr.addRow("Taille des matrices GEMM :", self.gemm)
        g.addLayout(fr)
        root.addWidget(gpu)

        disk = QGroupBox("Stockage")
        dl = QVBoxLayout(disk)
        dl.addWidget(QLabel("Disques testés (un fichier temporaire y est écrit puis supprimé) :"))
        self.drive_row = QHBoxLayout()
        self.drive_row.setSpacing(18)
        self.drive_boxes: dict[str, QCheckBox] = {}
        self.drive_hint = QLabel("Détection des disques…")
        self.drive_hint.setObjectName("muted")
        self.drive_row.addWidget(self.drive_hint)
        self.drive_row.addStretch(1)
        dl.addLayout(self.drive_row)
        fd = QFormLayout()
        fd.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.disk_size = QComboBox()
        for label, val in (("Selon la durée (512 Mo / 1 Go / 4 Go)", 0), ("256 Mo", 256), ("1 Go", 1024),
                           ("4 Go", 4096), ("8 Go", 8192)):
            self.disk_size.addItem(label, val)
        self.disk_size.setCurrentIndex(max(0, self.disk_size.findData(s.disk_size_mb)))
        self.disk_size.currentIndexChanged.connect(lambda _: self._set("disk_size_mb", self.disk_size.currentData()))
        fd.addRow("Taille du fichier de test :", self.disk_size)
        dl.addLayout(fd)
        root.addWidget(disk)

        mon = QGroupBox("Surveillance")
        fm = QFormLayout(mon)
        fm.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.interval = QSpinBox()
        self.interval.setRange(250, 10000)
        self.interval.setSingleStep(250)
        self.interval.setSuffix(" ms")
        self.interval.setValue(s.monitor_interval_ms)
        self.interval.valueChanged.connect(self._interval)
        fm.addRow("Rafraîchissement de la barre d'état :", self.interval)
        root.addWidget(mon)

        about = QGroupBox("À propos")
        al = QVBoxLayout(about)
        txt = QLabel(
            f"<b>TFLOPS Meter {__version__}</b><br>"
            "CPU : noyaux FMA AVX2 en C++ natif. RAM : débit multithread et latence par chaînage de pointeurs.<br>"
            "GPU NVIDIA : GEMM cuBLAS (Tensor Cores), débit VRAM via le pilote CUDA, capteurs NVML.<br>"
            "GPU AMD / Intel : crête FMA et débit mémoire via OpenCL.<br><br>"
            "Les TFLOPS « théoriques » des fiches techniques supposent la fréquence boost et des FMA parfaites ; "
            "une GEMM réelle atteint en général 70 à 90 % de cette valeur.")
        txt.setWordWrap(True)
        al.addWidget(txt)
        row = QHBoxLayout()
        b = QPushButton("Ouvrir le dossier des données")
        b.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(data_dir())))
        row.addWidget(b)
        row.addStretch(1)
        al.addLayout(row)
        root.addWidget(about)
        root.addStretch(1)

    def _set(self, key, value):
        setattr(self.main.settings, key, value)
        self.main.save_settings()

    def set_drives(self, drives: list[dict]):
        """Cases à cocher des lecteurs, une fois la détection terminée."""
        from ..hwdetails import drive_label
        from ..runner import default_disk_letters
        self.drive_hint.setVisible(not drives)
        self.drive_hint.setText("Aucun disque détecté")
        chosen = self.main.settings.disk_letters or default_disk_letters(drives)
        for i, d in enumerate(drives):
            c = QCheckBox(drive_label(d) + f" — {d['free_gb']:.0f} Go libres".replace(".", ","))
            c.setChecked(d["letter"] in chosen)
            c.toggled.connect(self._drives_changed)
            self.drive_row.insertWidget(i, c)
            self.drive_boxes[d["letter"]] = c

    def _drives_changed(self):
        self._set("disk_letters", [k for k, c in self.drive_boxes.items() if c.isChecked()])

    def _precisions(self):
        self._set("gpu_precisions", [p for p, c in self.prec.items() if c.isChecked()])

    def _interval(self, v):
        self._set("monitor_interval_ms", v)
        self.main.timer.setInterval(v)


# =========================================================================== fenêtre

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TFLOPS Meter")
        self.setWindowIcon(theme.app_icon())
        self.resize(1320, 860)
        self.settings = self.load_settings()
        self.history = History()
        self.sysinfo: dict = {}
        self.busy = False

        self.monitor = Monitor()
        self.tabs = QTabWidget()
        self.analysis = AnalysisTab(self)
        self.ai_tab = AiTab(self)
        self.catalog_tab = CatalogTab(self)
        self.image_tab = ImageCatalogTab(self)
        self.models_tabs = QTabWidget()  # « Modèles compatibles » : texte (Ollama) et images / vidéo
        self.models_tabs.setDocumentMode(True)
        self.models_tabs.addTab(self.catalog_tab, "💬  Texte (LLM)")
        self.models_tabs.addTab(self.image_tab, "🎨  Images et vidéo")
        self.stability = StabilityTab(self)
        self.history_tab = HistoryTab(self)
        self.settings_tab = SettingsTab(self)
        self.tabs.addTab(self.analysis, "Analyse")
        self.tabs.addTab(self.ai_tab, "IA locale")
        self.tabs.addTab(self.models_tabs, "Modèles compatibles")
        self.tabs.addTab(self.stability, "Stabilité")
        self.tabs.addTab(self.history_tab, "Historique")
        self.tabs.addTab(self.settings_tab, "Paramètres")
        self.setCentralWidget(self.tabs)
        self.analysis.saved.connect(lambda _: self.history_tab.reload())
        self.stability.saved.connect(lambda _: self.history_tab.reload())

        self.live = QLabel("Surveillance…")
        self.statusBar().addWidget(self.live, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(self.settings.monitor_interval_ms)
        self.timer.timeout.connect(self.refresh_live)
        self.timer.start()

        self.analysis.start_btn.setEnabled(False)
        self.stability.start_btn.setEnabled(False)
        self.info_worker = SysInfoWorker()
        self.info_worker.done.connect(self.on_sysinfo)
        self.info_worker.start()

    # ---- état
    def on_sysinfo(self, info):
        self.sysinfo = info
        self.analysis.show_sysinfo(info)
        self.analysis.start_btn.setEnabled(True)
        self.stability.start_btn.setEnabled(True)
        self.settings_tab.set_drives((info.get("details") or {}).get("drives") or [])
        self.ai_tab.refresh()

    def latest_measures(self) -> list[dict]:
        """Mesures les plus récentes de chaque test : analyse en cours, puis historique (plus récent d'abord)."""
        seen, out = set(), []

        def add(m: dict):
            key = (m["category"], m["device"], m["test"])  # par appareil : plusieurs disques / GPU
            if key not in seen and m.get("value") is not None and not m.get("error"):
                seen.add(key)
                out.append(m)
        for m in self.analysis.measures:
            add(m.to_dict())
        for r in self.history.runs(limit=30):
            if r["kind"] == "analyse":
                run = self.history.run(r["id"])
                for m in (run or {}).get("results", []):
                    add(m)
        return out

    # ---- image à partager
    def share_image(self):
        from .share_image import render
        if not self.sysinfo:
            return None
        return render(self.sysinfo, self.latest_measures())  # config + performances uniquement

    def save_share_image(self):
        img = self.share_image()
        if img is None:
            return
        pics = os.path.join(os.path.expanduser("~"), "Pictures")
        default = os.path.join(pics if os.path.isdir(pics) else os.path.expanduser("~"),
                               f"tflops_meter_{datetime.now():%Y-%m-%d_%H%M}.jpg")
        path, _ = QFileDialog.getSaveFileName(self, "Image pour Discord", default, "JPEG (*.jpg);;PNG (*.png)")
        if not path:
            return
        fmt_ = "PNG" if path.lower().endswith(".png") else "JPG"
        if img.save(path, fmt_, 92):
            self.statusBar().showMessage(f"Image enregistrée : {path}", 8000)
        else:
            QMessageBox.critical(self, "Image", f"Impossible d'enregistrer {path}")

    def copy_share_image(self):
        img = self.share_image()
        if img is not None:
            QGuiApplication.clipboard().setImage(img)
            self.statusBar().showMessage("Image copiée : collez-la dans Discord avec Ctrl+V", 8000)

    def can_start(self) -> bool:
        if self.busy:
            QMessageBox.information(self, "TFLOPS Meter", "Une mesure est déjà en cours.")
            return False
        return True

    def set_busy(self, busy: bool):
        self.busy = busy
        self.analysis.start_btn.setEnabled(not busy)
        self.stability.start_btn.setEnabled(not busy)
        self.ai_tab.quick_btn.setEnabled(not busy)
        if self.ai_tab.models_stack.currentWidget() is self.ai_tab.models_table:
            self.ai_tab.bench_btn.setEnabled(not busy)

    def refresh_live(self):
        try:
            s = self.monitor.sample()
        except Exception:
            return
        parts = []
        for g in s["gpus"]:
            if not g:
                continue
            p = [g["name"].replace("NVIDIA GeForce ", "")]
            if g["temp_c"] is not None:
                p.append(f"{g['temp_c']} °C")
            if g["clock_mhz"]:
                p.append(f"{g['clock_mhz']} MHz")
            if g["power_w"] is not None:
                p.append(f"{g['power_w']:.0f} W")
            if g["util_pct"] is not None:
                p.append(f"charge {g['util_pct']} %")
            if g["throttle"]:
                p.append("⚠ " + ", ".join(g["throttle"]))
            parts.append(" · ".join(p))
        c = s["cpu"]
        if c:
            cp = ["CPU"]
            if c.get("mhz"):
                cp.append(f"{c['mhz'] / 1000:.2f} GHz".replace(".", ","))
            if c.get("usage_pct") is not None:
                cp.append(f"charge {c['usage_pct']:.0f} %")
            parts.append(" · ".join(cp))
        self.live.setText("     │     ".join(parts) if parts else "Capteurs indisponibles")

    # ---- réglages
    def load_settings(self) -> Settings:
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                return Settings.from_dict(json.load(f))
        except (OSError, ValueError):
            return Settings()

    def save_settings(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings.to_dict(), f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    # ---- export
    def export_runs(self, ids: list[int], kind: str):
        if not ids:
            return
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        default = os.path.join(os.path.expanduser("~"), "Documents", f"tflops_meter_{stamp}.{kind}")
        filt = "JSON (*.json)" if kind == "json" else "CSV (*.csv)"
        path, _ = QFileDialog.getSaveFileName(self, "Exporter", default, filt)
        if not path:
            return
        runs = [r for r in (self.history.run(i) for i in ids) if r]
        try:
            (export_json if kind == "json" else export_csv)(runs, path)
        except OSError as e:
            QMessageBox.critical(self, "Export", str(e))
            return
        self.statusBar().showMessage(f"Exporté : {path}", 6000)

    def closeEvent(self, e):
        for w in (self.analysis.worker, self.stability.worker, self.ai_tab.quick, self.ai_tab.bench,
                  self.catalog_tab.updater, self.catalog_tab.puller, self.image_tab.updater):
            if w and w.isRunning():
                w.stop()
                w.wait(15000)
        for w in (self.info_worker, self.ai_tab.detect):  # détections courtes : on les laisse finir
            if w and w.isRunning():
                w.wait(30000)
        e.accept()
