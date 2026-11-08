"""Onglet « IA locale » : capacité du PC, modèles installés, mesure réelle, rapport pour IA, image Discord."""
from __future__ import annotations

import traceback

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QMessageBox, QPushButton, QScrollArea, QSplitter, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import ai_profile as ap
from .. import cuda, local_ai
from ..model import Measure, Settings
from . import theme
from .workers import AnalysisWorker

CONTEXTS = [4096, 8192, 16384, 32768, 65536]


class DetectWorker(QThread):
    done = Signal(dict)

    def run(self):
        res = {"tools": {}, "theo_vram": None}
        try:
            res["tools"] = local_ai.detect()
        except Exception:
            res["tools"] = {}
        try:
            devs = cuda.devices()
            if devs:
                res["theo_vram"] = cuda.theoretical_vram_gbs(max(devs, key=lambda d: d.memory))
        except Exception:
            pass
        self.done.emit(res)


class OllamaBenchWorker(QThread):
    result = Signal(dict)
    progress = Signal(int, int, str)
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(self, models: list[str], ctx: int):
        super().__init__()
        self.models, self.ctx = models, ctx
        self._cancel = False

    def stop(self):
        self._cancel = True

    def run(self):
        out = []
        for i, name in enumerate(self.models):
            if self._cancel:
                break
            self.progress.emit(i, len(self.models), name)
            try:
                r = local_ai.ollama_benchmark(name, self.ctx)
                local_ai.ollama_unload(name)  # libère la VRAM avant le modèle suivant
            except Exception as e:
                r = {"model": name, "error": f"{type(e).__name__} : {e}"}
            out.append(r)
            self.result.emit(r)
        self.finished_ok.emit(out)


def _item(text, align_right=False, color=None, bold=False, tip=None):
    it = QTableWidgetItem(text)
    if align_right:
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    if color:
        it.setForeground(QColor(color))
    if bold:
        f = it.font()
        f.setBold(True)
        it.setFont(f)
    if tip:
        it.setToolTip(tip)
    return it


def tps_color(t):
    if t is None:
        return theme.MUTED
    return theme.GOOD if t >= 30 else theme.ACCENT if t >= 15 else theme.WARN if t >= 7 else theme.BAD


class AiTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.tools: dict = {}
        self.theo_vram = None
        self.hw: ap.Hardware | None = None
        self.installed = []
        self.measured: dict[str, dict] = self._load_measured()
        self.bench: OllamaBenchWorker | None = None
        self.quick: AnalysisWorker | None = None
        self.detect: DetectWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Contexte visé :"))
        self.ctx = QComboBox()
        for c in CONTEXTS:
            self.ctx.addItem(f"{c // 1024}k tokens", c)
        self.ctx.setCurrentIndex(max(0, self.ctx.findData(getattr(main.settings, "ai_ctx", 8192))))
        self.ctx.setToolTip("Taille de la conversation / des documents gardés en mémoire par le modèle.\n"
                            "Plus elle est grande, plus le cache KV occupe de VRAM.")
        self.ctx.currentIndexChanged.connect(self._ctx_changed)
        bar.addWidget(self.ctx)
        self.refresh_btn = QPushButton("Actualiser")
        self.refresh_btn.setToolTip("Détecter à nouveau Ollama, LM Studio, ComfyUI et l'occupation de la VRAM")
        self.refresh_btn.clicked.connect(self.refresh)
        bar.addWidget(self.refresh_btn)
        self.quick_btn = QPushButton("⚡ Mesure rapide pour l'IA")
        self.quick_btn.setToolTip("Mesure en ≈ 15 s la RAM, la VRAM et la puissance FP16 du GPU,\n"
                                  "pour des estimations basées sur ce PC.")
        self.quick_btn.clicked.connect(self.quick_measure)
        bar.addWidget(self.quick_btn)
        self.status = QLabel("")
        self.status.setObjectName("muted")
        bar.addWidget(self.status, 1)
        root.addLayout(bar)

        self.banner = QLabel()
        self.banner.setObjectName("warning")
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        root.addWidget(self.banner)

        from .main_window import Card  # import local : évite la dépendance circulaire
        cards = QHBoxLayout()
        cards.setSpacing(12)
        self.c_vram = Card("Mémoire pour l'IA", theme.CATEGORY_COLORS["VRAM"])
        self.c_best = Card("Plus gros modèle fluide", theme.CATEGORY_COLORS["GPU"])
        self.c_speed = Card("Modèle 7-8B (Q4)", theme.ACCENT)
        self.c_tools = Card("Outils d'IA détectés", theme.CATEGORY_COLORS["RAM"])
        for c in (self.c_vram, self.c_best, self.c_speed, self.c_tools):
            cards.addWidget(c)
        root.addLayout(cards)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(10)

        self.models_box = QGroupBox("Vos modèles (Ollama)")
        mb = QVBoxLayout(self.models_box)
        self.models_stack = QStackedWidget()
        self.models_table = QTableWidget()
        self.models_table.setColumnCount(9)
        self.models_table.setHorizontalHeaderLabels(
            ["Modèle", "Taille", "Quant.", "Placement estimé", "Contexte max 100 % GPU", "Estimé",
             "Mesuré", "Lecture du prompt", "En VRAM (réel)"])
        self.models_table.verticalHeader().setVisible(False)
        self.models_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.models_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.models_table.setAlternatingRowColors(True)
        self.models_stack.addWidget(self.models_table)
        self.no_models = QLabel()
        self.no_models.setWordWrap(True)
        self.no_models.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.no_models.setTextFormat(Qt.RichText)
        self.no_models.setOpenExternalLinks(True)
        self.no_models.setStyleSheet("padding: 8px;")
        self.models_stack.addWidget(self.no_models)
        mb.addWidget(self.models_stack)
        row = QHBoxLayout()
        self.bench_btn = QPushButton("▶  Mesurer la vitesse réelle des modèles cochés")
        self.bench_btn.clicked.connect(self.start_bench)
        self.bench_stop = QPushButton("■  Arrêter")
        self.bench_stop.setObjectName("danger")
        self.bench_stop.setVisible(False)
        self.bench_stop.clicked.connect(lambda: self.bench and self.bench.stop())
        row.addWidget(self.bench_btn)
        row.addWidget(self.bench_stop)
        note = QLabel("Charge chaque modèle dans Ollama, génère 256 tokens puis le décharge.")
        note.setObjectName("muted")
        row.addWidget(note, 1)
        mb.addLayout(row)
        ll.addWidget(self.models_box, 3)

        cap = QGroupBox("Ce que ce PC peut faire tourner (estimation, tokens/s en génération)")
        cl = QVBoxLayout(cap)
        self.matrix = QTableWidget()
        self.matrix.setColumnCount(4)
        self.matrix.setHorizontalHeaderLabels(["Taille du modèle", "Q4_K_M (compact)", "Q5_K_M (équilibré)",
                                               "Q8_0 (qualité max)"])
        self.matrix.verticalHeader().setVisible(False)
        self.matrix.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.matrix.setSelectionMode(QAbstractItemView.NoSelection)
        self.matrix.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        cl.addWidget(self.matrix)
        legend = QLabel(f"<span style='color:{theme.GOOD}'>■</span> très fluide ≥ 30 &nbsp; "
                        f"<span style='color:{theme.ACCENT}'>■</span> fluide ≥ 15 &nbsp; "
                        f"<span style='color:{theme.WARN}'>■</span> utilisable ≥ 7 &nbsp; "
                        f"<span style='color:{theme.BAD}'>■</span> lent &nbsp;·&nbsp; "
                        "« mixte » = une partie du modèle déborde en RAM")
        legend.setObjectName("muted")
        cl.addWidget(legend)
        ll.addWidget(cap, 3)
        split.addWidget(left)

        right = QGroupBox("Constats et conseils")
        rl = QVBoxLayout(right)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.advice = QLabel()
        self.advice.setWordWrap(True)
        self.advice.setTextFormat(Qt.RichText)
        self.advice.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.advice.setOpenExternalLinks(True)
        self.advice.setTextInteractionFlags(Qt.TextBrowserInteraction)
        scroll.setWidget(self.advice)
        rl.addWidget(scroll)
        split.addWidget(right)
        split.setSizes([900, 420])
        root.addWidget(split, 1)

        out = QHBoxLayout()
        lab = QLabel("Partager :")
        lab.setObjectName("muted")
        out.addWidget(lab)
        self.copy_report = QPushButton("📋  Copier le rapport pour Claude / ChatGPT")
        self.copy_report.setObjectName("primary")
        self.copy_report.setStyleSheet("padding: 8px 16px; font-size: 10.5pt;")
        self.copy_report.setToolTip("Copie un rapport complet (matériel, mesures, modèles, constats) et une demande "
                                    "toute prête : collez-le dans la conversation avec l'IA.")
        self.copy_report.clicked.connect(self.copy_ai_report)
        out.addWidget(self.copy_report)
        b = QPushButton("Enregistrer le rapport (.md)")
        b.clicked.connect(self.save_ai_report)
        out.addWidget(b)
        out.addStretch(1)
        b = QPushButton("🖼  Image pour Discord (JPG)")
        b.clicked.connect(self.main.save_share_image)
        out.addWidget(b)
        b = QPushButton("Copier l'image")
        b.setToolTip("Copie l'image dans le presse-papiers : collez-la directement dans Discord (Ctrl+V)")
        b.clicked.connect(self.main.copy_share_image)
        out.addWidget(b)
        root.addLayout(out)

    # ------------------------------------------------------------------ données
    def _load_measured(self) -> dict:
        """Dernière mesure réelle connue de chaque modèle (historique)."""
        out = {}
        for r in reversed(self.main.history.runs()):
            if r["kind"] == "ia":
                for name, b in (r["summary"].get("bench") or {}).items():
                    out[name] = b
        return out

    def ctx_value(self) -> int:
        return self.ctx.currentData() or 8192

    def _ctx_changed(self):
        self.main.settings.ai_ctx = self.ctx_value()
        self.main.save_settings()
        self.recompute()

    def refresh(self):
        if self.detect and self.detect.isRunning():
            return
        self.status.setText("Détection des outils d'IA…")
        self.refresh_btn.setEnabled(False)
        self.detect = DetectWorker()
        self.detect.done.connect(self._on_detect)
        self.detect.start()

    def _on_detect(self, res):
        self.tools = res.get("tools") or {}
        self.theo_vram = res.get("theo_vram")
        self.refresh_btn.setEnabled(True)
        self.status.setText("")
        self.recompute()

    def recompute(self):
        if not self.main.sysinfo:
            return
        try:
            gpu_sample = None
            nv = self.main.monitor.nvml
            if nv.available:
                gpu_sample = nv.sample(0)
            self.hw = ap.build_hardware(self.main.sysinfo, self.main.latest_measures(), gpu_sample,
                                        nv.driver_version() if nv.available else None, self.theo_vram)
            ctx = self.ctx_value()
            models = (self.tools.get("ollama") or {}).get("models") or []
            self.installed = ap.installed_estimates(self.hw, models, ctx)
            self._fill_cards(ctx)
            self._fill_models()
            self._fill_matrix(ctx)
            self._fill_advice(ctx)
            if hasattr(self.main, "catalog_tab"):
                self.main.catalog_tab.refresh_table()
            if hasattr(self.main, "image_tab"):
                self.main.image_tab.refresh_table()
        except Exception:
            self.advice.setText(f"<pre>{traceback.format_exc(limit=4)}</pre>")

    # ------------------------------------------------------------------ affichage
    def _fill_cards(self, ctx):
        hw = self.hw
        if hw.vram_gb:
            self.c_vram.set(f"{ap.fr(hw.usable_gb)} Go", f"VRAM utilisable · {ap.fr(hw.vram_used_now_gb)} Go occupés "
                                                         "en ce moment")
        else:
            self.c_vram.set(f"{ap.fr(hw.ram_gb, 0)} Go", "RAM (pas de VRAM dédiée) · calcul sur le processeur")
        matrix = dict(ap.class_matrix(hw, ctx))
        names = [n for n, *_ in ap.MODEL_CLASSES]
        best = None
        for n in names:
            for q in ap.QUANT_BITS:
                e = matrix[n][q]
                if (e.gen_tps or 0) >= 15 and (not best or e.total_gb > best[2].total_gb):
                    best = (n, q, e)
        if best:
            n, q, e = best
            self.c_best.set(n.split(" (")[0], f"{q} · ≈ {ap.fr(e.gen_tps, 0)} tok/s · {e.placement}")
        else:
            self.c_best.set("3-4B", "même les petits modèles restent lents sur ce PC")
        e8 = matrix["7-8B"]["Q4_K_M"]
        m8 = next((b for name, b in self.measured.items() if b.get("gen_tps") and
                   any(m["name"] == name and 6 <= (ap.parse_params(m.get("params", "")) or 0) <= 9
                       for m, _ in self.installed)), None)
        if m8:
            self.c_speed.set(f"{ap.fr(m8['gen_tps'], 0)} tok/s", f"mesuré ({m8['model']})")
        else:
            self.c_speed.set(f"≈ {ap.fr(e8.gen_tps, 0)} tok/s", f"estimé · {e8.placement} · "
                                                              f"{'mesures réelles' if hw.measured else 'débits non mesurés'}")
        t = self.tools
        found = []
        o = t.get("ollama", {})
        if o.get("running"):
            found.append(f"Ollama {o.get('version', '')} ({len(o.get('models') or [])} modèles)")
        elif o.get("installed"):
            found.append("Ollama (arrêté)")
        if t.get("lmstudio", {}).get("running"):
            found.append("LM Studio")
        elif t.get("lmstudio", {}).get("installed"):
            found.append("LM Studio (installé)")
        if t.get("comfyui", {}).get("running"):
            found.append("ComfyUI")
        if found:
            self.c_tools.set(str(len(found)), " · ".join(found))
        else:
            self.c_tools.set("Aucun", "voir « Pour commencer » à droite")
        msgs = []
        if not hw.measured:
            msgs.append("Estimations provisoires : cliquez sur « ⚡ Mesure rapide pour l'IA » pour les baser sur "
                        "les vrais débits de ce PC.")
        if hw.vram_gb and hw.vram_used_now_gb > 1.5:
            msgs.append(f"{ap.fr(hw.vram_used_now_gb)} Go de VRAM déjà occupés par d'autres applications"
                        + (" (ComfyUI ouvert)" if t.get("comfyui", {}).get("running") else "")
                        + " : les modèles qui tiennent juste risquent de déborder en RAM.")
        self.banner.setText("⚠  " + "\n⚠  ".join(msgs) if msgs else "")
        self.banner.setVisible(bool(msgs))

    def _fill_models(self):
        o = self.tools.get("ollama", {})
        if not o.get("running") or not o.get("models"):
            self.models_box.setTitle("Vos modèles")
            if o.get("running"):
                txt = ("Ollama fonctionne mais aucun modèle n'est installé.<br>Exemple : "
                       "<code>ollama pull qwen3:8b</code> dans un terminal, puis « Actualiser ».")
            elif o.get("installed"):
                txt = "Ollama est installé mais arrêté : lancez-le puis cliquez sur « Actualiser »."
            else:
                txt = ("Aucun outil d'IA locale détecté sur ce PC — ce n'est pas un problème : le tableau "
                       "ci-dessous montre ce que ce PC <b>pourrait</b> faire tourner.<br><br>"
                       "Pour essayer : <a style='color:#4f8cff' href='https://ollama.com'>Ollama</a> "
                       "(simple, en ligne de commande) ou "
                       "<a style='color:#4f8cff' href='https://lmstudio.ai'>LM Studio</a> "
                       "(interface graphique). Revenez ensuite ici pour mesurer la vitesse réelle de vos modèles.")
            self.no_models.setText(txt)
            self.models_stack.setCurrentWidget(self.no_models)
            self.bench_btn.setEnabled(False)
            return
        self.models_box.setTitle(f"Vos modèles (Ollama {o.get('version', '')})")
        self.models_stack.setCurrentWidget(self.models_table)
        self.bench_btn.setEnabled(not self.main.busy)
        t = self.models_table
        checked = {t.item(r, 0).text() for r in range(t.rowCount())
                   if t.item(r, 0) and t.item(r, 0).checkState() == Qt.Checked}
        t.setRowCount(len(self.installed))
        for r, (m, e) in enumerate(self.installed):
            name = QTableWidgetItem(m["name"])
            if e is not None:
                name.setFlags(name.flags() | Qt.ItemIsUserCheckable)
                name.setCheckState(Qt.Checked if (m["name"] in checked or not checked) else Qt.Unchecked)
            moe = f"MoE : {m['experts_used']} experts actifs sur {m['experts']}\n" if m.get("experts") else ""
            name.setToolTip(f"{moe}{m.get('params', '')} paramètres · contexte max du modèle : "
                            f"{m.get('max_ctx') or '?'} tokens")
            t.setItem(r, 0, name)
            t.setItem(r, 1, _item(f"{ap.fr(m['size_bytes'] / 1e9)} Go", True))
            t.setItem(r, 2, _item(m.get("quant", "")))
            b = self.measured.get(m["name"], {})
            if e is None:
                for c, txt in ((3, "embeddings (pas de génération)"), (4, ""), (5, ""), (6, ""), (7, ""), (8, "")):
                    t.setItem(r, c, _item(txt, color=theme.MUTED))
                continue
            pcol = theme.GOOD if e.gpu_frac >= 0.999 else theme.WARN if e.possible else theme.BAD
            t.setItem(r, 3, _item(e.placement, color=pcol,
                                  tip=f"Poids {ap.fr(e.weights_gb)} Go + cache KV {ap.fr(e.kv_gb)} Go "
                                      f"+ tampons ≈ {ap.fr(e.total_gb)} Go"))
            t.setItem(r, 4, _item(f"{e.max_ctx_gpu // 1024}k" if e.max_ctx_gpu and e.max_ctx_gpu >= 1024 else "—",
                                  True))
            t.setItem(r, 5, _item(f"≈ {ap.fr(e.gen_tps, 0)} tok/s" if e.gen_tps else "—", True,
                                  tps_color(e.gen_tps)))
            if b.get("error"):
                t.setItem(r, 6, _item("échec", True, theme.BAD, tip=b["error"]))
                t.setItem(r, 7, _item(""))
                t.setItem(r, 8, _item(""))
            else:
                t.setItem(r, 6, _item(f"{ap.fr(b['gen_tps'])} tok/s" if b.get("gen_tps") else "—", True,
                                      tps_color(b.get("gen_tps")), bold=True,
                                      tip=f"contexte {b.get('num_ctx')} · chargement {ap.fr(b.get('load_s'))} s"
                                      if b else None))
                t.setItem(r, 7, _item(f"{ap.fr(b['prompt_tps'], 0)} tok/s" if b.get("prompt_tps") else "—", True))
                g = b.get("gpu_pct")
                t.setItem(r, 8, _item(f"{ap.fr(g, 0)} %" if g is not None else "—", True,
                                      theme.GOOD if g and g >= 99.5 else theme.WARN if g is not None else None))
        h = t.horizontalHeader()
        for c in range(t.columnCount()):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)  # noms de modèles en entier
        h.setSectionResizeMode(3, QHeaderView.Stretch)

    def _fill_matrix(self, ctx):
        rows = ap.class_matrix(self.hw, ctx)
        self.matrix.setRowCount(len(rows))
        for r, (name, row) in enumerate(rows):
            self.matrix.setItem(r, 0, _item(name, bold=True))
            for c, q in enumerate(ap.QUANT_BITS, start=1):
                e = row[q]
                if not e.possible:
                    it = _item(f"trop gros ({ap.fr(e.total_gb, 0)} Go)", color=theme.MUTED)
                else:
                    place = "" if e.gpu_frac >= 0.999 else (" · CPU" if e.gpu_frac <= 0.001 else " · mixte")
                    it = _item(f"≈ {ap.fr(e.gen_tps, 0)} tok/s{place}", color=tps_color(e.gen_tps))
                it.setToolTip(f"{ap.fr(e.total_gb)} Go nécessaires à {ctx // 1024}k de contexte · {e.placement}"
                              + (f" · lecture du prompt ≈ {ap.fr(e.prefill_tps, 0)} tok/s" if e.prefill_tps else ""))
                it.setTextAlignment(Qt.AlignCenter)
                self.matrix.setItem(r, c, it)
        self.matrix.resizeRowsToContents()

    def _fill_advice(self, ctx):
        recs = ap.recommendations(self.hw, ctx, self.tools, self.installed)
        start = ap.getting_started(self.hw, self.tools)
        main = [r for r in recs if r not in start]
        html = "".join(f"<p style='margin:0 0 9px 0'>• {r}</p>" for r in main)
        if start:
            html += f"<p style='color:{theme.ACCENT};font-weight:600;margin:12px 0 6px 0'>Pour commencer</p>"
            html += "".join(f"<p style='margin:0 0 9px 0'>• {r}</p>" for r in start)
        html += (f"<p style='color:{theme.MUTED};font-weight:600;margin:12px 0 6px 0'>Génération d'images et de "
                 "vidéos (liste complète : Modèles compatibles › Images et vidéo)</p>")
        from .image_tab import LEVEL_COLORS
        for name, v, level in ap.image_capacity(self.hw):
            html += (f"<p style='margin:0 0 4px 0'>{name} : "
                     f"<span style='color:{LEVEL_COLORS[level]}'>{v}</span></p>")
        self.advice.setText(html)

    # ------------------------------------------------------------------ mesures
    def quick_measure(self):
        if not self.main.can_start():
            return
        cfg = Settings(preset="Rapide", do_cpu=False, do_gpu=True, do_ram=True, do_vram=True,
                       gpu_precisions=["FP16"], gpu_fma_peak=False)
        self.quick = AnalysisWorker(cfg)
        self.quick.progress.connect(lambda f, label: self.status.setText(f"Mesure : {label} ({f * 100:.0f} %)"))
        self.quick.finished_ok.connect(self._quick_done)
        self.quick.failed.connect(lambda e: (self.main.set_busy(False), QMessageBox.critical(self, "Mesure", e)))
        self.main.set_busy(True)
        self.quick.start()

    def _quick_done(self, results, cancelled):
        self.main.set_busy(False)
        if results:
            run_id = self.main.history.save_run("analyse", "Mesure IA", self.main.sysinfo, results,
                                                summary={"highlights": "Mesure rapide pour l'IA (RAM, VRAM, FP16)"})
            self.main.history_tab.reload()
        self.status.setText("Mesure terminée." if not cancelled else "Mesure arrêtée.")
        self.recompute()

    def start_bench(self):
        if not self.main.can_start():
            return
        t = self.models_table
        names = [t.item(r, 0).text() for r in range(t.rowCount())
                 if t.item(r, 0) and t.item(r, 0).flags() & Qt.ItemIsUserCheckable
                 and t.item(r, 0).checkState() == Qt.Checked]
        if not names:
            QMessageBox.information(self, "IA locale", "Cochez au moins un modèle.")
            return
        self.bench = OllamaBenchWorker(names, self.ctx_value())
        self.bench.progress.connect(lambda i, n, name: self.status.setText(f"Modèle {i + 1}/{n} : {name}…"))
        self.bench.result.connect(self._bench_result)
        self.bench.finished_ok.connect(self._bench_done)
        self.main.set_busy(True)
        self.bench_btn.setVisible(False)
        self.bench_stop.setVisible(True)
        self.bench.start()

    def _bench_result(self, r):
        self.measured[r["model"]] = r
        self._fill_models()
        self._fill_cards(self.ctx_value())

    def _bench_done(self, results):
        self.main.set_busy(False)
        self.bench_btn.setVisible(True)
        self.bench_stop.setVisible(False)
        self.status.setText(f"{len(results)} modèle(s) mesuré(s).")
        ok = [r for r in results if not r.get("error")]
        if ok:
            est = {m["name"]: e for m, e in self.installed}
            measures = []
            for r in ok:
                e = est.get(r["model"])
                measures.append(Measure(
                    "IA", r["model"], "Génération", r.get("gen_tps"), "tok/s",
                    theoretical=e.gen_tps if e else None,
                    details=f"prompt {ap.fr(r.get('prompt_tps'), 0)} tok/s · en VRAM {ap.fr(r.get('gpu_pct'), 0)} % "
                            f"· contexte {r.get('num_ctx')}"))
            best = max(ok, key=lambda r: r.get("gen_tps") or 0)
            self.main.history.save_run(
                "ia", f"contexte {self.ctx_value() // 1024}k", self.main.sysinfo, measures,
                summary={"highlights": f"{len(ok)} modèle(s) · meilleur {best['model']} "
                                       f"{ap.fr(best.get('gen_tps'))} tok/s",
                         "bench": {r["model"]: r for r in ok}})
            self.main.history_tab.reload()
        self.recompute()

    # ------------------------------------------------------------------ partage
    def report_text(self) -> str:
        if self.hw is None:
            self.recompute()
        return ap.build_report(self.hw, self.tools, self.ctx_value(), self.installed, self.measured,
                               self.main.latest_measures())

    def copy_ai_report(self):
        if not self.main.sysinfo:
            return
        QGuiApplication.clipboard().setText(self.report_text())
        self.status.setText("Rapport copié : collez-le dans Claude ou ChatGPT, puis précisez vos usages.")

    def save_ai_report(self):
        if not self.main.sysinfo:
            return
        import os
        from datetime import datetime
        default = os.path.join(os.path.expanduser("~"), "Documents",
                               f"profil_ia_{datetime.now():%Y-%m-%d_%H%M}.md")
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le rapport", default, "Markdown (*.md)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.report_text())
            self.status.setText(f"Rapport enregistré : {path}")

    def share_data(self) -> dict | None:
        """Résumé IA pour l'image Discord."""
        if self.hw is None:
            return None
        ctx = self.ctx_value()
        head = []
        matrix = dict(ap.class_matrix(self.hw, ctx))
        if self.hw.vram_gb:
            fits = [n for n, *_ in ap.MODEL_CLASSES if not n.startswith("MoE")
                    and matrix[n]["Q4_K_M"].gpu_frac >= 0.999]
            if fits:
                e = matrix[fits[-1]]["Q4_K_M"]
                head.append(f"Plus gros modèle 100 % GPU ({ctx // 1024}k de contexte) : {fits[-1]} en Q4 "
                            f"(≈ {ap.fr(e.gen_tps, 0)} tok/s)")
        else:
            ok = [n for n, *_ in ap.MODEL_CLASSES if (matrix[n]["Q4_K_M"].gen_tps or 0) >= 7]
            n = ok[-1] if ok else "3-4B"
            head.append(f"Sans GPU dédié : jusqu'à la classe {n} en Q4 "
                        f"(≈ {ap.fr(matrix[n]['Q4_K_M'].gen_tps, 0)} tok/s sur le processeur)")
        e8 = matrix["7-8B"]["Q4_K_M"]
        head.append(f"Modèle 7-8B en Q4 : ≈ {ap.fr(e8.gen_tps, 0)} tok/s ({e8.placement})")
        if not ap.has_ai_tools(self.tools):
            head.append("Aucun outil d'IA locale installé pour l'instant")
        if not self.hw.measured:
            head.append("Estimations provisoires (débits non mesurés)")
        imgs = [n for n, _v, level in ap.image_capacity(self.hw) if level >= 2]
        if imgs:
            head.append("Images : " + ", ".join(imgs[-2:]) + " utilisables")
        models = []
        for m, e in self.installed:
            if e is None:
                continue
            b = self.measured.get(m["name"])
            if b and b.get("gen_tps"):
                models.append((m["name"], b["gen_tps"], b.get("gpu_pct"), True))
        if not models:
            models = [(m["name"], e.gen_tps, e.gpu_frac * 100, False) for m, e in self.installed if e and e.gen_tps]
        models.sort(key=lambda x: -(x[1] or 0))
        return {"headline": head, "models": models[:7]}
