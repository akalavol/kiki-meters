"""Carte « config + performances » en image (JPG / presse-papiers) à partager sur Discord."""
from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen

from .. import __version__
from ..ai_profile import fr
from . import theme

W = 1200
PAD = 40


def _font(size, bold=False):
    f = QFont("Segoe UI")
    f.setPixelSize(size)
    f.setBold(bold)
    return f


def _box(p: QPainter, r: QRectF, fill=theme.PANEL, border=theme.BORDER, radius=14):
    path = QPainterPath()
    path.addRoundedRect(r, radius, radius)
    p.fillPath(path, QColor(fill))
    if border:
        p.setPen(QPen(QColor(border), 1.5))
        p.drawPath(path)


def _text(p, x, y, w, h, text, size, color, bold=False, align=Qt.AlignLeft | Qt.AlignVCenter):
    p.setFont(_font(size, bold))
    p.setPen(QColor(color))
    p.drawText(QRectF(x, y, w, h), int(align), text)


def _find(measures, cat, test_part, device=None):
    for m in measures:
        if (m.get("category") == cat and test_part in m.get("test", "") and m.get("value")
                and not m.get("error") and (device is None or m.get("device") == device)):
            return m
    return None


def _size(gb):
    return f"{fr(gb / 1000, 1)} To" if gb >= 1000 else f"{fr(gb, 0)} Go"


def config_lines(sysinfo: dict) -> list[tuple[str, str]]:
    det = sysinfo.get("details") or {}
    lines = [("CPU", f"{sysinfo.get('cpu', '?')} · {sysinfo.get('cores') or '?'} cœurs / "
                     f"{sysinfo.get('threads') or '?'} threads")]
    gpus = sysinfo.get("gpus") or []
    dedicated = [g for g in gpus if not g.get("integrated")]
    for g in dedicated:
        lines.append(("GPU", f"{g['name']} · {fr(g.get('memory_gb'), 0)} Go de VRAM"))
    if not dedicated:
        lines.append(("GPU", f"{gpus[0]['name']} (intégré)" if gpus else "aucune carte graphique dédiée"))
    ram = det.get("ram")
    if ram:
        lines.append(("RAM", f"{fr(ram['total_gb'], 0)} Go {ram['type']} · {ram['layout']} · {ram['speed_mts']} MT/s"))
    else:
        lines.append(("RAM", f"{fr(sysinfo.get('ram_gb'), 0)} Go"))
    if det.get("board"):
        lines.append(("CM", det["board"]))
    for d in det.get("disks") or []:
        if d.get("kind") == "USB":
            continue
        letters = ", ".join(x + ":" for x in d.get("letters") or [])
        lines.append(("SSD" if "SSD" in d["kind"] else "HDD" if d["kind"] == "HDD" else "Disque",
                      f"{d['model']} · {_size(d['size_gb'])} · {d['kind']}" + (f" · {letters}" if letters else "")))
    if det.get("os"):
        lines.append(("OS", det["os"]))
    return lines


LABEL_COLORS = {"CPU": theme.CATEGORY_COLORS["CPU"], "GPU": theme.CATEGORY_COLORS["GPU"],
                "RAM": theme.CATEGORY_COLORS["RAM"], "SSD": theme.CATEGORY_COLORS["Disque"],
                "HDD": theme.CATEGORY_COLORS["Disque"], "Disque": theme.CATEGORY_COLORS["Disque"]}


def render(sysinfo: dict, measures: list[dict], _ai=None) -> QImage:
    """measures : dicts au format Measure (dernières mesures connues)."""
    img = QImage(W, 3000, QImage.Format_RGB32)
    img.fill(QColor(theme.BG))
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)

    # ---- en-tête
    g = QLinearGradient(0, 0, W, 0)
    g.setColorAt(0, QColor("#2f5fd0"))
    g.setColorAt(1, QColor("#1f9e72"))
    p.fillRect(0, 0, W, 104, g)
    _text(p, PAD, 16, 700, 46, "TFLOPS Meter", 38, "white", True)
    _text(p, PAD, 60, 800, 28, "Configuration et performances mesurées", 18, "#e6eeff")
    _text(p, W - PAD - 400, 26, 400, 30, datetime.now().strftime("%d/%m/%Y"), 20, "white", True,
          Qt.AlignRight | Qt.AlignVCenter)
    _text(p, W - PAD - 400, 58, 400, 24, f"version {__version__}", 14, "#dbe6ff", False,
          Qt.AlignRight | Qt.AlignVCenter)
    y = 126

    # ---- configuration
    lines = config_lines(sysinfo)
    h = 56 + 33 * len(lines)
    _box(p, QRectF(PAD, y, W - 2 * PAD, h))
    _text(p, PAD + 22, y + 12, 400, 30, "Configuration", 19, theme.TEXT, True)
    yy = y + 48
    for label, txt in lines:
        _text(p, PAD + 22, yy, 80, 33, label, 15, LABEL_COLORS.get(label, theme.MUTED), True)
        _text(p, PAD + 100, yy, W - 2 * PAD - 120, 33, txt, 18, theme.TEXT)
        yy += 33
    y += h + 20

    # ---- tuiles de performances
    tiles = []
    for cat, part, title, unit, dec in (
        ("GPU", "GEMM FP16", "GPU FP16 (Tensor)", "TFLOPS", 1), ("GPU", "GEMM FP32", "GPU FP32", "TFLOPS", 1),
        ("VRAM", "Bande passante", "VRAM", "Go/s", 0), ("CPU", "FP32 multicœur", "CPU multicœur", "GFLOPS", 0),
        ("CPU", "FP32 monocœur", "CPU monocœur", "GFLOPS", 0), ("RAM", "Lecture", "RAM lecture", "Go/s", 1),
        ("RAM", "Latence", "RAM latence", "ns", 0),
    ):
        m = _find(measures, cat, part)
        if m:
            tiles.append((cat, title, fr(m["value"], dec), unit))
    if not _find(measures, "GPU", "GEMM"):  # GPU sans cuBLAS : crête FMA
        m = _find(measures, "GPU", "Crête FMA")
        if m:
            tiles.insert(0, ("GPU", "GPU " + m["test"].replace("Crête FMA ", "crête "), fr(m["value"], 1), "TFLOPS"))
    seq = [m for m in measures if m.get("category") == "Disque" and m.get("test") == "Lecture séquentielle"
           and m.get("value") and not m.get("error")]
    if seq:
        best = max(seq, key=lambda m: m["value"])
        tiles.append(("Disque", "Disque lecture séq.", fr(best["value"], 0), "Mo/s"))
    if tiles:
        cols = 4
        tw = (W - 2 * PAD - (cols - 1) * 16) / cols
        th = 112
        for i, (cat, title, val, unit) in enumerate(tiles[:8]):
            cx = PAD + (i % cols) * (tw + 16)
            cy = y + (i // cols) * (th + 16)
            color = theme.CATEGORY_COLORS.get(cat, theme.ACCENT)
            _box(p, QRectF(cx, cy, tw, th))
            p.fillRect(QRectF(cx, cy + 18, 5, th - 36), QColor(color))
            _text(p, cx + 22, cy + 10, tw - 30, 26, title.upper(), 13, theme.MUTED, True)
            _text(p, cx + 22, cy + 36, tw - 30, 48, val, 36, color, True)
            _text(p, cx + 22, cy + 80, tw - 30, 22, unit, 15, theme.MUTED)
        y += ((min(len(tiles), 8) + cols - 1) // cols) * (th + 16) + 4
    else:
        _box(p, QRectF(PAD, y, W - 2 * PAD, 70))
        _text(p, PAD, y, W - 2 * PAD, 70, "Aucune mesure : lancez une analyse dans TFLOPS Meter", 18, theme.MUTED,
              False, Qt.AlignCenter)
        y += 90

    # ---- barres GPU (toutes les précisions)
    gpu = [m for m in measures if m.get("category") == "GPU" and m.get("value") and not m.get("error")]
    if len(gpu) > 1:
        h = 52 + 36 * len(gpu)
        _box(p, QRectF(PAD, y, W - 2 * PAD, h))
        _text(p, PAD + 22, y + 12, 600, 30, "Calcul GPU (TFLOPS)", 19, theme.TEXT, True)
        top = max(max(m["value"], m.get("theoretical") or 0) for m in gpu)
        bx, bw = PAD + 250, W - 2 * PAD - 250 - 110
        by = y + 50
        for m in gpu:
            _text(p, PAD + 22, by, 225, 28, m["test"].replace("GEMM ", ""), 16, theme.TEXT)
            if m.get("theoretical"):
                _box(p, QRectF(bx, by + 6, bw * m["theoretical"] / top, 16), theme.BORDER, None, 5)
            _box(p, QRectF(bx, by + 6, max(4, bw * m["value"] / top), 16), theme.CATEGORY_COLORS["GPU"], None, 5)
            _text(p, bx + bw + 12, by, 100, 28, fr(m["value"], 2), 16, theme.TEXT, True)
            by += 36
        y += h + 20

    # ---- stockage : un disque par ligne
    devices = []
    for m in measures:
        if m.get("category") == "Disque" and m.get("device") not in devices:
            devices.append(m["device"])
    if devices:
        cols = [("Lecture séquentielle", "Lecture séq."), ("Écriture séquentielle", "Écriture séq."),
                ("Lecture aléatoire 4K (Q1)", "4K lecture"), ("Écriture aléatoire 4K (Q1)", "4K écriture"),
                ("Lecture aléatoire 4K (Q32)", "4K Q32")]
        h = 96 + 38 * len(devices)
        _box(p, QRectF(PAD, y, W - 2 * PAD, h))
        _text(p, PAD + 22, y + 12, 600, 30, "Stockage (Mo/s)", 19, theme.TEXT, True)
        name_w = 400
        cw = (W - 2 * PAD - 44 - name_w) / len(cols)
        hy = y + 50
        for j, (_t, short) in enumerate(cols):
            _text(p, PAD + 22 + name_w + j * cw, hy, cw, 26, short.upper(), 12, theme.MUTED, True,
                  Qt.AlignRight | Qt.AlignVCenter)
        ry = hy + 32
        for dev in devices:
            short_dev = dev.replace(" (SSD NVMe)", " · NVMe").replace(" (SSD)", " · SSD").replace(" (HDD)", " · HDD")
            _text(p, PAD + 22, ry, name_w - 10, 34, short_dev, 16, theme.TEXT)
            for j, (test, _s) in enumerate(cols):
                m = _find(measures, "Disque", test, dev)
                if m and m.get("test") == test:
                    _text(p, PAD + 22 + name_w + j * cw, ry, cw, 34, fr(m["value"], 0), 17,
                          theme.CATEGORY_COLORS["Disque"] if j < 2 else theme.TEXT, j < 2,
                          Qt.AlignRight | Qt.AlignVCenter)
                else:
                    _text(p, PAD + 22 + name_w + j * cw, ry, cw, 34, "—", 17, theme.MUTED, False,
                          Qt.AlignRight | Qt.AlignVCenter)
            ry += 38
        y += h + 20

    _text(p, PAD, y, W - 2 * PAD, 26, "Mesuré avec TFLOPS Meter", 14, theme.MUTED, False, Qt.AlignCenter)
    y += 44
    p.end()
    return img.copy(0, 0, W, int(y))
