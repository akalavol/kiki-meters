"""Onglet « Modèles compatibles » : catalogue Ollama classé selon ce que ce PC peut faire tourner."""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMessageBox, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from .. import ai_profile as ap
from .. import catalog, local_ai
from . import theme


class UpdateWorker(QThread):
    progress = Signal(int, int, str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self._cancel = False

    def stop(self):
        self._cancel = True

    def run(self):
        try:
            self.done.emit(catalog.update(self.progress.emit, lambda: self._cancel))
        except Exception as e:
            self.failed.emit(str(e))


class PullWorker(QThread):
    progress = Signal(float, str)
    done = Signal(bool, str)

    def __init__(self, tag: str):
        super().__init__()
        self.tag = tag
        self._cancel = False

    def stop(self):
        self._cancel = True

    def run(self):
        req = urllib.request.Request(local_ai.OLLAMA + "/api/pull", method="POST",
                                     data=json.dumps({"model": self.tag, "stream": True}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                for line in r:
                    if self._cancel:
                        self.done.emit(False, "Téléchargement interrompu")
                        return
                    msg = json.loads(line.decode("utf-8") or "{}")
                    if msg.get("error"):
                        self.done.emit(False, msg["error"])
                        return
                    total, comp = msg.get("total"), msg.get("completed")
                    frac = comp / total if total and comp else -1
                    txt = msg.get("status", "")
                    if total and comp:
                        txt += f" — {comp / 1e9:.1f} / {total / 1e9:.1f} Go"
                    self.progress.emit(frac, txt)
            self.done.emit(True, f"{self.tag} installé")
        except Exception as e:
            self.done.emit(False, f"{type(e).__name__} : {e}")


class SortItem(QTableWidgetItem):
    """Cellule triée sur une valeur numérique (stockée dans UserRole)."""

    def __lt__(self, other):
        a, b = self.data(Qt.UserRole), other.data(Qt.UserRole)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a < b
        return super().__lt__(other)


def _cell(text, value=None, color=None, bold=False, right=False, tip=None):
    it = SortItem(text)
    it.setData(Qt.UserRole, value if value is not None else text.lower())
    if color:
        it.setForeground(QColor(color))
    if bold:
        f = it.font()
        f.setBold(True)
        it.setFont(f)
    if right:
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    return it


def _pulls(v):
    if not v:
        return "—"
    return f"{v / 1e6:.1f} M".replace(".", ",") if v >= 1e6 else f"{v / 1e3:.0f} k"


class CatalogTab(QWidget):
    COLS = ["Modèle", "Usages", "Taille", "Contexte max", "Placement", "Vitesse estimée", "Verdict",
            "Popularité", "Installé"]

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.cat = catalog.load()
        self.items: list[dict] = []
        self.shown: list[dict] = []
        self.updater: UpdateWorker | None = None
        self.puller: PullWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        intro = QLabel("Tous les modèles de la bibliothèque Ollama, évalués pour <b>ce PC</b> : mémoire nécessaire, "
                       "placement en VRAM ou en RAM et vitesse de génération estimée. Le contexte visé se règle "
                       "dans l'onglet IA locale.")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        root.addWidget(intro)

        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Rechercher (qwen, llama, code, vision…)")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.apply_filters)
        bar.addWidget(self.search, 2)
        self.usage = QComboBox()
        self.usage.addItem("Tous les usages", "")
        for u in catalog.USAGES:
            self.usage.addItem(u, u)
        self.usage.currentIndexChanged.connect(self.apply_filters)
        bar.addWidget(self.usage)
        self.only_good = QCheckBox("Seulement ceux qui tournent bien")
        self.only_good.setToolTip("Vitesse estimée ≥ 15 tokens/s sur ce PC")
        self.only_good.setChecked(True)
        self.only_good.toggled.connect(self.apply_filters)
        bar.addWidget(self.only_good)
        self.best_only = QCheckBox("Une taille par modèle")
        self.best_only.setToolTip("Pour chaque modèle, la plus grosse variante qui tourne bien sur ce PC")
        self.best_only.setChecked(True)
        self.best_only.toggled.connect(self.apply_filters)
        bar.addWidget(self.best_only)
        bar.addStretch(1)
        self.update_btn = QPushButton("⟳  Mettre à jour la liste")
        self.update_btn.setToolTip("Télécharge la liste actuelle des modèles depuis ollama.com (≈ 15 s)")
        self.update_btn.clicked.connect(self.start_update)
        bar.addWidget(self.update_btn)
        root.addLayout(bar)

        info = QHBoxLayout()
        self.source = QLabel()
        self.source.setObjectName("muted")
        info.addWidget(self.source, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedWidth(260)
        self.progress.setVisible(False)
        info.addWidget(self.progress)
        root.addLayout(info)

        self.table = QTableWidget()
        self.table.setColumnCount(len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self.on_select)
        self.table.doubleClicked.connect(lambda _: self.copy_command())
        root.addWidget(self.table, 1)

        self.detail = QLabel("Sélectionnez un modèle.")
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.RichText)
        self.detail.setMinimumHeight(54)
        root.addWidget(self.detail)

        act = QHBoxLayout()
        self.install_btn = QPushButton("⬇  Installer avec Ollama")
        self.install_btn.setObjectName("primary")
        self.install_btn.setStyleSheet("padding: 8px 18px; font-size: 10.5pt;")
        self.install_btn.clicked.connect(self.install)
        act.addWidget(self.install_btn)
        self.cancel_pull = QPushButton("■  Annuler")
        self.cancel_pull.setObjectName("danger")
        self.cancel_pull.setVisible(False)
        self.cancel_pull.clicked.connect(lambda: self.puller and self.puller.stop())
        act.addWidget(self.cancel_pull)
        self.copy_btn = QPushButton("Copier la commande")
        self.copy_btn.clicked.connect(self.copy_command)
        act.addWidget(self.copy_btn)
        self.page_btn = QPushButton("Page du modèle")
        self.page_btn.clicked.connect(lambda: self._open(f"https://ollama.com/library/{self._sel()['tag']}"))
        act.addWidget(self.page_btn)
        self.hf_btn = QPushButton("Version GGUF (LM Studio)")
        self.hf_btn.setToolTip("Rechercher ce modèle au format GGUF sur Hugging Face, pour LM Studio ou llama.cpp")
        self.hf_btn.clicked.connect(lambda: self._open(
            f"https://huggingface.co/models?library=gguf&search={self._sel()['family']}"))
        act.addWidget(self.hf_btn)
        self.pull_status = QLabel("")
        self.pull_status.setObjectName("muted")
        act.addWidget(self.pull_status, 1)
        root.addLayout(act)
        self._update_source()
        self.on_select()

    # ------------------------------------------------------------------ données
    def _update_source(self):
        f = self.cat.get("fetched")
        when = datetime.fromisoformat(f).strftime("%d/%m/%Y") if f else "?"
        n = len(self.cat.get("models", []))
        old = ""
        if f and (datetime.now() - datetime.fromisoformat(f)).days > 30:
            old = " — plus d'un mois : pensez à mettre à jour"
        self.source.setText(f"Liste du {when} · {n} modèles · {self.cat.get('origin', '')}{old}")

    def installed_names(self) -> set[str]:
        models = (self.main.ai_tab.tools.get("ollama") or {}).get("models") or []
        names = set()
        for m in models:
            n = m["name"]
            names.add(n)
            if n.endswith(":latest"):
                names.add(n[:-7])
        return names

    def refresh_table(self):
        hw = self.main.ai_tab.hw
        if hw is None:
            return
        self.items = catalog.entries(self.cat, hw, self.main.ai_tab.ctx_value())
        self.apply_filters()

    def apply_filters(self):
        if not self.items:
            return
        q = self.search.text().strip().lower()
        usage = self.usage.currentData()
        items = self.items
        if self.best_only.isChecked():
            items = catalog.best_per_family(items)
        if self.only_good.isChecked():
            items = [i for i in items if i["estimate"].possible and
                     ("Embeddings" in i["usages"] or (i["estimate"].gen_tps or 0) >= 15)]
        if usage:
            items = [i for i in items if usage in i["usages"]]
        if q:
            items = [i for i in items if q in i["tag"].lower() or q in i["description"].lower()
                     or any(q in u.lower() for u in i["usages"])]
        self.shown = items
        self._fill(items)

    def _fill(self, items):
        t = self.table
        t.setSortingEnabled(False)
        t.setRowCount(len(items))
        installed = self.installed_names()
        for r, i in enumerate(items):
            e = i["estimate"]
            emb = "Embeddings" in i["usages"]
            name = _cell(i["tag"], i["tag"], bold=True, tip=i["description"])
            name.setData(Qt.UserRole + 1, r)
            t.setItem(r, 0, name)
            moe = f" · MoE {i['active_b']:g}B actifs" if i["moe"] else ""
            t.setItem(r, 1, _cell(", ".join(i["usages"]) + moe, None))
            approx = " (estimée)" if i["estimated_size"] else ""
            t.setItem(r, 2, _cell(f"{ap.fr(i['size_gb'])} Go{approx}", i["size_gb"], right=True))
            ctx = i["ctx"]
            ctx_txt = "—" if not ctx else f"{ctx // 1000}k" if ctx >= 1000 else str(ctx)
            t.setItem(r, 3, _cell(ctx_txt, ctx or 0, right=True))
            pcol = theme.GOOD if e.gpu_frac >= 0.999 else theme.WARN if e.possible else theme.BAD
            t.setItem(r, 4, _cell(e.placement, e.gpu_frac if e.possible else -1, pcol,
                                  tip=f"{ap.fr(e.total_gb)} Go nécessaires (poids + cache KV + tampons)"))
            if emb:
                t.setItem(r, 5, _cell("—", 0, right=True))
                t.setItem(r, 6, _cell("léger (recherche / RAG)", 50, theme.GOOD))
            else:
                from .ai_tab import tps_color
                tps = e.gen_tps or 0
                t.setItem(r, 5, _cell(f"≈ {ap.fr(tps, 0)} tok/s" if e.possible else "—", tps,
                                      tps_color(e.gen_tps), bold=True, right=True))
                t.setItem(r, 6, _cell(e.verdict, tps, tps_color(e.gen_tps)))
            t.setItem(r, 7, _cell(_pulls(i["pulls"]), i["pulls"], right=True,
                                  tip=f"Téléchargements · mis à jour : {i.get('updated') or '?'}"))
            inst = i["tag"] in installed
            t.setItem(r, 8, _cell("✔" if inst else "", 1 if inst else 0, theme.GOOD))
        t.setSortingEnabled(True)
        if not getattr(self, "_sorted_once", False):  # tri initial : les plus populaires d'abord
            t.sortItems(7, Qt.DescendingOrder)
            self._sorted_once = True
        h = t.horizontalHeader()
        for c in range(len(self.COLS)):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        self.on_select()

    def _sel(self) -> dict | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        idx = self.table.item(rows[0].row(), 0).data(Qt.UserRole + 1)
        return self.shown[idx] if idx is not None and idx < len(self.shown) else None

    def on_select(self):
        s = self._sel()
        ollama = (self.main.ai_tab.tools.get("ollama") or {}).get("running")
        busy = self.puller is not None and self.puller.isRunning()
        for b in (self.copy_btn, self.page_btn, self.hf_btn):
            b.setEnabled(s is not None)
        self.install_btn.setEnabled(bool(s and ollama and not busy))
        self.install_btn.setToolTip("" if ollama else "Ollama n'est pas lancé (ou pas installé : ollama.com)")
        if not s:
            self.detail.setText("Sélectionnez un modèle pour voir sa description et la commande d'installation.")
            return
        e = s["estimate"]
        line = (f"<b>{s['tag']}</b> — {s['description']}<br>"
                f"<span style='color:{theme.MUTED}'>{ap.fr(s['size_gb'])} Go à télécharger · "
                f"{ap.fr(e.total_gb)} Go en mémoire à {self.main.ai_tab.ctx_value() // 1024}k de contexte · "
                f"{e.placement}")
        if e.max_ctx_gpu:
            line += f" · contexte max en 100 % GPU ≈ {e.max_ctx_gpu // 1000}k"
        line += f" · commande : <code>ollama pull {s['tag']}</code></span>"
        self.detail.setText(line)

    # ------------------------------------------------------------------ actions
    def _open(self, url):
        QDesktopServices.openUrl(QUrl(url))

    def copy_command(self):
        s = self._sel()
        if s:
            QGuiApplication.clipboard().setText(f"ollama pull {s['tag']}")
            self.pull_status.setText(f"Commande copiée : ollama pull {s['tag']}")

    def start_update(self):
        if self.updater and self.updater.isRunning():
            self.updater.stop()
            return
        self.updater = UpdateWorker()
        self.updater.progress.connect(self._upd_progress)
        self.updater.done.connect(self._upd_done)
        self.updater.failed.connect(self._upd_failed)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.update_btn.setText("■  Annuler la mise à jour")
        self.updater.start()

    def _upd_progress(self, done, total, name):
        self.progress.setValue(int(done / max(total, 1) * 1000))
        self.source.setText(f"Mise à jour : {done}/{total} · {name}")

    def _upd_done(self, cat):
        self.cat = catalog.load()
        self._upd_end()
        self.refresh_table()
        self.pull_status.setText(f"Liste mise à jour : {len(cat['models'])} modèles.")

    def _upd_failed(self, msg):
        self._upd_end()
        QMessageBox.warning(self, "Mise à jour", msg + "\n\nLa liste précédente reste utilisée.")

    def _upd_end(self):
        self.progress.setVisible(False)
        self.update_btn.setText("⟳  Mettre à jour la liste")
        self._update_source()

    def install(self):
        s = self._sel()
        if not s:
            return
        e = s["estimate"]
        if not e.possible or (e.gen_tps or 0) < 5:
            if QMessageBox.question(self, "Installer", f"{s['tag']} sera {e.verdict} sur ce PC "
                                    f"({e.placement}). L'installer quand même ?") != QMessageBox.Yes:
                return
        if QMessageBox.question(self, "Installer", f"Télécharger {s['tag']} ({ap.fr(s['size_gb'])} Go) "
                                "avec Ollama ?") != QMessageBox.Yes:
            return
        self.puller = PullWorker(s["tag"])
        self.puller.progress.connect(self._pull_progress)
        self.puller.done.connect(self._pull_done)
        self.cancel_pull.setVisible(True)
        self.install_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.puller.start()

    def _pull_progress(self, frac, txt):
        if frac >= 0:
            self.progress.setValue(int(frac * 1000))
        self.pull_status.setText(txt)

    def _pull_done(self, ok, msg):
        self.cancel_pull.setVisible(False)
        self.progress.setVisible(False)
        self.pull_status.setText(("✔ " if ok else "✖ ") + msg)
        self.on_select()
        if ok:
            self.main.ai_tab.refresh()  # le nouveau modèle apparaît dans « Vos modèles »
