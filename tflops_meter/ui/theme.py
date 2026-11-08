"""Thème sombre de l'application."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPixmap, QLinearGradient

ACCENT = "#4f8cff"
GOOD = "#3ccf91"
WARN = "#f5a524"
BAD = "#f05252"
BG = "#14161b"
PANEL = "#1c1f26"
PANEL2 = "#232733"
BORDER = "#2e3340"
TEXT = "#e6e8ee"
MUTED = "#8b93a7"

CATEGORY_COLORS = {"CPU": "#4f8cff", "GPU": "#3ccf91", "RAM": "#f5a524", "VRAM": "#b07cff", "Disque": "#ff6f91",
                   "IA": "#b07cff", "Stabilité": "#3ccf91"}

STYLESHEET = f"""
QWidget {{ color: {TEXT}; font-size: 10pt; }}
QMainWindow, QDialog {{ background: {BG}; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 10px 18px; border: none;
               border-bottom: 2px solid transparent; font-weight: 600; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QGroupBox {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; margin-top: 14px;
             padding: 14px 12px 10px 12px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {MUTED}; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; }}
QLabel#cardTitle {{ color: {MUTED}; font-size: 9pt; font-weight: 600; }}
QLabel#cardValue {{ font-size: 20pt; font-weight: 700; }}
QLabel#cardSub {{ color: {MUTED}; font-size: 8.5pt; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#warning {{ background: #3a2d12; color: {WARN}; border: 1px solid #6b5216; border-radius: 8px; padding: 8px; }}
QLabel#verdict {{ font-size: 16pt; font-weight: 700; }}
QPushButton {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 7px 14px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #5c6375; border-color: {PANEL2}; }}
QPushButton#primary {{ background: {ACCENT}; border: none; color: white; font-size: 12pt; font-weight: 700;
                      padding: 12px; }}
QPushButton#primary:hover {{ background: #6a9dff; }}
QPushButton#primary:disabled {{ background: #2c3b5c; color: #8ea2c9; }}
QPushButton#danger {{ background: #4a1f24; border: 1px solid #7a2a31; color: #ffb4b4; }}
QProgressBar {{ background: {PANEL2}; border: none; border-radius: 6px; height: 12px; text-align: center;
                color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}
QTreeWidget, QTableWidget {{ background: {PANEL}; alternate-background-color: #20242d; border: 1px solid {BORDER};
                             border-radius: 10px; gridline-color: {BORDER}; selection-background-color: #2b3d63; }}
QHeaderView::section {{ background: {PANEL2}; color: {MUTED}; border: none; padding: 6px; font-weight: 600; }}
QComboBox, QSpinBox {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 8px; }}
QComboBox QAbstractItemView {{ background: {PANEL2}; selection-background-color: #2b3d63; }}
QCheckBox {{ spacing: 8px; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QScrollArea {{ border: none; }}
QToolTip {{ background: {PANEL2}; color: {TEXT}; border: 1px solid {BORDER}; }}
"""


def apply(app):
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in {
        QPalette.Window: BG, QPalette.WindowText: TEXT, QPalette.Base: PANEL, QPalette.AlternateBase: "#20242d",
        QPalette.Text: TEXT, QPalette.Button: PANEL2, QPalette.ButtonText: TEXT, QPalette.Highlight: "#2b3d63",
        QPalette.HighlightedText: TEXT, QPalette.ToolTipBase: PANEL2, QPalette.ToolTipText: TEXT,
        QPalette.PlaceholderText: MUTED,
    }.items():
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    f = QFont("Segoe UI", 10)
    app.setFont(f)
    app.setStyleSheet(STYLESHEET)


def app_icon() -> QIcon:
    icon = QIcon()
    for size in (16, 32, 48, 64, 128, 256):
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        g = QLinearGradient(0, 0, size, size)
        g.setColorAt(0, QColor("#4f8cff"))
        g.setColorAt(1, QColor("#3ccf91"))
        p.setBrush(g)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 0, size, size, size * 0.22, size * 0.22)
        # trois barres de « graphique »
        p.setBrush(QColor("white"))
        w = size * 0.16
        for i, h in enumerate((0.35, 0.55, 0.75)):
            x = size * (0.2 + i * 0.22)
            p.drawRoundedRect(int(x), int(size * (0.85 - h)), int(w), int(size * h), w * 0.3, w * 0.3)
        p.end()
        icon.addPixmap(pm)
    return icon
