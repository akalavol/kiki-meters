"""Sous-onglet « Images et vidéo » : modèles de génération (diffusion) évalués pour ce PC."""
from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMessageBox, QProgressBar, QPushButton, QTableWidget, QVBoxLayout, QWidget)

from .. import ai_profile as ap
from .. import image_models as im
from . import theme
from .catalog_tab import _cell

LEVEL_COLORS = {4: theme.GOOD, 3: theme.ACCENT, 2: theme.WARN, 1: theme.BAD, 0: theme.BAD, -1: theme.MUTED}
KINDS = [("Images et vidéos", ""), ("Images (texte → image)", "Texte → image"),
         ("Édition d'image", "Édition d'image"), ("Vidéo", "vidéo"), ("Anime / illustration", "Anime / illustration"),
         ("Photo réaliste", "Photo réaliste")]
COMFY_DOWNLOAD = "https://www.comfy.org/download"


class ImageUpdateWorker(QThread):
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
            self.done.emit(im.update(self.progress.emit, lambda: self._cancel))
        except Exception as e:
            self.failed.emit(str(e))


def _downloads(v):
    if not v:
        return "—"
    return f"{v / 1e6:.1f} M".replace(".", ",") if v >= 1e6 else f"{v / 1e3:.0f} k"


class ImageCatalogTab(QWidget):
    COLS = ["Modèle", "Usages", "Format", "Taille", "Placement", "Temps estimé", "Verdict", "Popularité",
            "Installé"]

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.cat = im.load()
        self.items: list[dict] = []
        self.shown: list[dict] = []
        self.updater: ImageUpdateWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        intro = QLabel("Modèles de génération d'images et de vidéos les plus téléchargés sur Hugging Face, évalués "
                       "pour <b>ce PC</b> : VRAM nécessaire et temps estimé par image (1024×1024) ou par clip "
                       "de 5 s, dans ComfyUI, Forge ou SwarmUI. Les estimations sont basées sur la puissance FP16 "
                       "mesurée du GPU.")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        root.addWidget(intro)

        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Rechercher (flux, qwen, wan, sdxl, anime…)")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.apply_filters)
        bar.addWidget(self.search, 2)
        self.kind = QComboBox()
        for label, data in KINDS:
            self.kind.addItem(label, data)
        self.kind.currentIndexChanged.connect(self.apply_filters)
        bar.addWidget(self.kind)
        self.only_good = QCheckBox("Seulement ceux qui tournent bien")
        self.only_good.setToolTip("Image en 90 s maximum, ou clip vidéo en 10 min maximum, sur ce PC")
        self.only_good.setChecked(True)
        self.only_good.toggled.connect(self.apply_filters)
        bar.addWidget(self.only_good)
        self.best_only = QCheckBox("Un format par modèle")
        self.best_only.setToolTip("Pour chaque modèle, le format le plus fidèle qui tient en VRAM (BF16, FP8 ou GGUF)")
        self.best_only.setChecked(True)
        self.best_only.toggled.connect(self.apply_filters)
        bar.addWidget(self.best_only)
        bar.addStretch(1)
        self.update_btn = QPushButton("⟳  Mettre à jour la liste")
        self.update_btn.setToolTip("Télécharge la liste actuelle des modèles depuis Hugging Face (quelques secondes)")
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
        self.table.doubleClicked.connect(lambda _: self._open_page())
        root.addWidget(self.table, 1)

        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.RichText)
        self.detail.setMinimumHeight(70)
        root.addWidget(self.detail)

        act = QHBoxLayout()
        self.page_btn = QPushButton("Page du modèle")
        self.page_btn.setObjectName("primary")
        self.page_btn.setStyleSheet("padding: 8px 18px; font-size: 10.5pt;")
        self.page_btn.clicked.connect(self._open_page)
        act.addWidget(self.page_btn)
        self.comfy_btn = QPushButton("Fichiers pour ComfyUI")
        self.comfy_btn.setToolTip("Fichiers prêts à l'emploi (modèle, encodeur de texte, VAE) au format ComfyUI")
        self.comfy_btn.clicked.connect(self._open_comfy_files)
        act.addWidget(self.comfy_btn)
        self.gguf_btn = QPushButton("Version GGUF")
        self.gguf_btn.setToolTip("Versions compressées (Q8, Q5, Q4) pour les cartes avec moins de VRAM. Dans ComfyUI : "
                                 "extension ComfyUI-GGUF, fichiers dans models/unet")
        self.gguf_btn.clicked.connect(self._open_gguf)
        act.addWidget(self.gguf_btn)
        self.copy_btn = QPushButton("Copier le nom")
        self.copy_btn.clicked.connect(self._copy_name)
        act.addWidget(self.copy_btn)
        self.app_btn = QPushButton()
        self.app_btn.clicked.connect(self._open_app)
        act.addWidget(self.app_btn)
        self.status = QLabel("")
        self.status.setObjectName("muted")
        act.addWidget(self.status, 1)
        root.addLayout(act)
        self._update_source()
        self.on_select()

    # ------------------------------------------------------------------ données
    def _comfy(self) -> dict:
        return (self.main.ai_tab.tools or {}).get("comfyui") or {}

    def _update_source(self, extra=""):
        f = self.cat.get("fetched")
        when = datetime.fromisoformat(f).strftime("%d/%m/%Y") if f else "?"
        n = len(self.cat.get("families", []))
        old = ""
        if f and (datetime.now() - datetime.fromisoformat(f)).days > 30:
            old = " — plus d'un mois : pensez à mettre à jour"
        txt = f"Liste du {when} · {n} modèles · {self.cat.get('origin', '')}{old}"
        hw = self.main.ai_tab.hw
        if hw is not None and not im.gpu_tflops(hw)[1] and hw.vram_gb:
            txt += " · ⚠ puissance GPU non mesurée : lancez « ⚡ Mesure rapide pour l'IA » (onglet IA locale)"
        if hw is not None and hw.gpu and not im.is_nvidia(hw):
            txt += " · GPU non NVIDIA : vitesse très variable selon le logiciel (ZLUDA, DirectML, ROCm)"
        self.source.setText(txt + extra)

    def refresh_table(self):
        hw = self.main.ai_tab.hw
        if hw is None:
            return
        files = self._comfy().get("model_files") or []
        self.items = im.entries(self.cat, hw, im.installed_keys(self.cat, files))
        self._update_source()
        self.apply_filters()

    def apply_filters(self):
        if not self.items:
            return
        q = self.search.text().strip().lower()
        kind = self.kind.currentData()
        items = self.items
        if self.best_only.isChecked():
            items = im.best_per_family(items)
        if self.only_good.isChecked():
            items = [i for i in items if i["estimate"].runs_well]
        if kind == "vidéo":
            items = [i for i in items if i["profile"]["video"]]
        elif kind:
            items = [i for i in items if kind in i["usages"]]
        if q:
            items = [i for i in items if q in i["family"]["id"].lower() or q in i["description"].lower()
                     or any(q in u.lower() for u in i["usages"]) or q in i["profile"]["label"].lower()]
        self.shown = items
        self._fill(items)

    def _fill(self, items):
        t = self.table
        t.setSortingEnabled(False)
        t.setRowCount(len(items))
        for r, i in enumerate(items):
            e = i["estimate"]
            name = _cell(i["name"], i["name"].lower(), bold=True, tip=f"{i['family']['id']}\n{i['description']}")
            name.setData(Qt.UserRole + 1, r)
            t.setItem(r, 0, name)
            t.setItem(r, 1, _cell(", ".join(i["usages"]), None))
            t.setItem(r, 2, _cell(i["format"], im.QUALITY.get(i["format"], 0)))
            exp = " (2 experts)" if i["profile"].get("experts", 1) > 1 else ""
            t.setItem(r, 3, _cell(f"{ap.fr(i['size_gb'])} Go{exp}", i["size_gb"], right=True,
                                  tip=f"{ap.fr(i['params_b'])} milliards de paramètres"))
            pcol = theme.GOOD if e.possible and not e.cpu and e.gpu_frac >= 0.999 else \
                theme.WARN if e.possible else theme.BAD
            t.setItem(r, 4, _cell(e.placement, e.gpu_frac if e.possible else -1, pcol,
                                  tip=f"≈ {ap.fr(e.vram_need_gb)} Go de VRAM pour tout garder sur le GPU"))
            unit = "clip" if e.video else "image"
            col = LEVEL_COLORS[e.level]
            t.setItem(r, 5, _cell(f"≈ {im.duration(e.total_s)} / {unit}" if e.possible else "—",
                                  e.total_s if e.total_s is not None else 1e12, col, bold=True, right=True))
            t.setItem(r, 6, _cell(e.verdict, e.level, col))
            t.setItem(r, 7, _cell(_downloads(i["downloads"]), i["downloads"], right=True,
                                  tip="Téléchargements sur Hugging Face ces 30 derniers jours, copies et versions "
                                      "compressées comprises"))
            t.setItem(r, 8, _cell("✔" if i["installed"] else "", 1 if i["installed"] else 0, theme.GOOD,
                                  tip="Trouvé dans les dossiers de modèles de ComfyUI" if i["installed"] else None))
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
        for b in (self.page_btn, self.comfy_btn, self.gguf_btn, self.copy_btn):
            b.setEnabled(s is not None)
        running = self._comfy().get("running")
        self.app_btn.setText("Ouvrir ComfyUI" if running else "Télécharger ComfyUI")
        self.app_btn.setToolTip("" if running else "ComfyUI : le logiciel de référence pour ces modèles (gratuit)")
        if not s:
            self.detail.setText("Sélectionnez un modèle pour voir les fichiers nécessaires et le détail de l'estimation.")
            return
        e, prof, fam = s["estimate"], s["profile"], s["family"]
        te_gb = prof["te_gb"] or 5.0
        files = f"modèle {s['format']} {ap.fr(s['size_gb'])} Go"
        if not prof.get("checkpoint"):
            files += f" + encodeur de texte ({prof['te']}, ≈ {ap.fr(te_gb)} Go) + VAE"
        else:
            files += " (encodeur de texte et VAE inclus)"
        calc = (f"{prof['steps']} étapes{' avec guidage (CFG)' if prof['cfg'] else ''} · "
                f"≈ {ap.fr(e.step_s, 2)} s par étape" if e.possible else "")
        lic = f" · licence {fam['license']}" if fam.get("license") not in (None, "other", "unknown") else ""
        gated = " · accès à demander sur Hugging Face" if fam.get("gated") else ""
        folder = ("models/unet (extension ComfyUI-GGUF)" if s["format"].startswith("GGUF")
                  else "models/checkpoints" if prof.get("checkpoint") else "models/diffusion_models")
        line = (f"<b>{s['name']}</b> — {s['description']}<br>"
                f"<span style='color:{theme.MUTED}'>Fichiers : {files} · dossier ComfyUI : {folder}<br>"
                f"Référence : {prof['ref']} · {calc} · {e.placement} "
                f"(≈ {ap.fr(e.vram_need_gb)} Go de VRAM pour tout garder sur le GPU){lic}{gated}</span>")
        self.detail.setText(line)

    # ------------------------------------------------------------------ actions
    def _open(self, url):
        QDesktopServices.openUrl(QUrl(url))

    def _open_page(self):
        s = self._sel()
        if s:
            self._open(f"{im.HF}/{s['family']['id']}")

    def _open_comfy_files(self):
        s = self._sel()
        if not s:
            return
        repo = s["family"].get("comfy_repo")
        self._open(f"{im.HF}/{repo}" if repo else f"{im.HF}/models?search={s['name']}+comfy")

    def _open_gguf(self):
        s = self._sel()
        if not s:
            return
        repo = s["family"].get("gguf_repo")
        self._open(f"{im.HF}/{repo}" if repo else f"{im.HF}/models?search={s['name']}+gguf")

    def _copy_name(self):
        s = self._sel()
        if s:
            QGuiApplication.clipboard().setText(s["family"]["id"])
            self.status.setText(f"Copié : {s['family']['id']}")

    def _open_app(self):
        c = self._comfy()
        self._open(c.get("url") if c.get("running") else COMFY_DOWNLOAD)

    def start_update(self):
        if self.updater and self.updater.isRunning():
            self.updater.stop()
            return
        self.updater = ImageUpdateWorker()
        self.updater.progress.connect(self._upd_progress)
        self.updater.done.connect(self._upd_done)
        self.updater.failed.connect(self._upd_failed)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.update_btn.setText("■  Annuler la mise à jour")
        self.updater.start()

    def _upd_progress(self, done, total, name):
        self.progress.setValue(int(done / max(total, 1) * 1000))
        self.source.setText(f"Mise à jour : {name}")

    def _upd_done(self, cat):
        self.cat = im.load()
        self._upd_end()
        self.refresh_table()
        self.status.setText(f"Liste mise à jour : {len(cat['families'])} modèles.")

    def _upd_failed(self, msg):
        self._upd_end()
        QMessageBox.warning(self, "Mise à jour", msg + "\n\nLa liste précédente reste utilisée.")

    def _upd_end(self):
        self.progress.setVisible(False)
        self.update_btn.setText("⟳  Mettre à jour la liste")
        self._update_source()
