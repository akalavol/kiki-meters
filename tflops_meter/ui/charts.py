"""Graphiques (QtCharts) aux couleurs du thème."""
from __future__ import annotations

from PySide6.QtCharts import (QAbstractBarSeries, QBarCategoryAxis, QBarSet, QChart, QChartView,
                              QHorizontalBarSeries, QLineSeries, QValueAxis)
from PySide6.QtCore import QLocale, QMargins, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from . import theme


def _base_chart(title: str = "") -> QChart:
    c = QChart()
    c.setTheme(QChart.ChartThemeDark)
    c.setBackgroundBrush(QColor(theme.PANEL))
    c.setBackgroundRoundness(10)
    c.setPlotAreaBackgroundVisible(False)
    c.setMargins(QMargins(6, 6, 6, 6))
    c.legend().setLabelColor(QColor(theme.MUTED))
    if title:
        c.setTitle(title)
        f = QFont("Segoe UI", 10)
        f.setBold(True)
        c.setTitleFont(f)
        c.setTitleBrush(QColor(theme.TEXT))
    c.setAnimationOptions(QChart.NoAnimation)
    c.setLocalizeNumbers(True)  # virgule décimale française
    c.setLocale(QLocale(QLocale.French, QLocale.France))
    return c


def _nice(ax: QValueAxis, lo: float, hi: float):
    """Bornes et graduations rondes (0, 250, 500… plutôt que 280, 559…)."""
    ax.setRange(lo, hi if hi > lo else lo + 1)
    ax.applyNiceNumbers()
    span = ax.max() - ax.min()
    ax.setLabelFormat("%.0f" if span >= 10 else "%.1f" if span >= 1 else "%.2f")


def _style_axis(ax):
    ax.setLabelsColor(QColor(theme.MUTED))
    ax.setGridLineColor(QColor(theme.BORDER))
    ax.setLinePenColor(QColor(theme.BORDER))
    ax.setLabelsFont(QFont("Segoe UI", 8))


class BarChart(QChartView):
    """Barres horizontales : mesuré (et théorique si connu)."""

    def __init__(self, title: str, unit: str, color: str):
        super().__init__()
        self.unit, self.color = unit, color
        self.setRenderHint(QPainter.Antialiasing)
        self.setMinimumHeight(220)
        self.setChart(_base_chart(title))
        self.set_data([])

    def set_data(self, rows: list[tuple[str, float, float | None]]):
        c = self.chart()
        c.removeAllSeries()
        for ax in list(c.axes()):
            c.removeAxis(ax)
        if not rows:
            c.legend().hide()
            return
        measured = QBarSet("Mesuré")
        measured.setColor(QColor(self.color))
        measured.setBorderColor(QColor(self.color))
        measured.setLabelColor(QColor(theme.TEXT))
        theo = QBarSet("Théorique")
        theo.setColor(QColor(theme.BORDER))
        theo.setBorderColor(QColor(theme.BORDER))
        theo.setLabelColor(QColor(0, 0, 0, 0))  # étiquette seulement sur la valeur mesurée
        has_theo = any(t for _, _, t in rows)
        cats = []
        for label, v, t in reversed(rows):  # le premier en haut
            cats.append(label)
            measured.append(v or 0)
            theo.append(t or 0)
        series = QHorizontalBarSeries()
        if has_theo:
            series.append(theo)
        series.append(measured)
        series.setLabelsVisible(True)
        series.setLabelsFormat("@value")
        series.setLabelsPrecision(4)
        series.setLabelsPosition(QAbstractBarSeries.LabelsOutsideEnd)
        series.setBarWidth(0.75)
        c.addSeries(series)
        ay = QBarCategoryAxis()
        ay.append(cats)
        ax = QValueAxis()
        ax.setTitleText(self.unit)
        ax.setTitleBrush(QColor(theme.MUTED))
        top = max(max(v or 0, t or 0) for _, v, t in rows)
        _nice(ax, 0, top * 1.15 if top else 1)  # marge pour les étiquettes à droite des barres
        for a in (ax, ay):
            _style_axis(a)
        c.addAxis(ay, Qt.AlignLeft)
        c.addAxis(ax, Qt.AlignBottom)
        series.attachAxis(ay)
        series.attachAxis(ax)
        c.legend().setVisible(has_theo)
        c.legend().setAlignment(Qt.AlignBottom)


class LiveChart(QChartView):
    """Deux courbes en temps réel sur deux axes verticaux (gauche / droite)."""

    def __init__(self, title: str, left: tuple[str, str], right: tuple[str, str]):
        super().__init__()
        self.setRenderHint(QPainter.Antialiasing)
        self.setMinimumHeight(220)
        c = _base_chart(title)
        self.setChart(c)
        self.s_left = QLineSeries(name=left[0])
        self.s_right = QLineSeries(name=right[0])
        self.s_left.setPen(QPen(QColor(left[1]), 2.2))
        self.s_right.setPen(QPen(QColor(right[1]), 1.6, Qt.DashLine))
        c.addSeries(self.s_left)
        c.addSeries(self.s_right)
        self.ax = QValueAxis(); self.ax.setTitleText("secondes"); self.ax.setLabelFormat("%.0f")
        self.ay_l = QValueAxis(); self.ay_l.setTitleText(left[0])
        self.ay_r = QValueAxis(); self.ay_r.setTitleText(right[0])
        for a in (self.ax, self.ay_l, self.ay_r):
            _style_axis(a)
            a.setTitleBrush(QColor(theme.MUTED))
        self.ay_l.setLabelFormat("%.1f")
        self.ay_r.setLabelFormat("%.0f")
        c.addAxis(self.ax, Qt.AlignBottom)
        c.addAxis(self.ay_l, Qt.AlignLeft)
        c.addAxis(self.ay_r, Qt.AlignRight)
        self.s_left.attachAxis(self.ax); self.s_left.attachAxis(self.ay_l)
        self.s_right.attachAxis(self.ax); self.s_right.attachAxis(self.ay_r)
        c.legend().setAlignment(Qt.AlignBottom)
        self.clear()

    def set_titles(self, left: str, right: str):
        self.s_left.setName(left); self.ay_l.setTitleText(left)
        self.s_right.setName(right); self.ay_r.setTitleText(right)

    def clear(self):
        self.s_left.clear()
        self.s_right.clear()
        self._lmax = self._rmax = 0.0
        self._rmin = None
        self.ax.setRange(0, 60)
        self.ay_l.setRange(0, 1)
        self.ay_r.setRange(0, 100)

    def add(self, t: float, left: float | None, right: float | None):
        if left is not None:
            self.s_left.append(t, left)
            if left > self._lmax:
                self._lmax = left
                _nice(self.ay_l, 0, left * 1.12)
        if right is not None:
            self.s_right.append(t, right)
            changed = right > self._rmax or self._rmin is None or right < self._rmin
            self._rmax = max(self._rmax, right)
            self._rmin = right if self._rmin is None else min(self._rmin, right)
            if changed:
                _nice(self.ay_r, max(0.0, self._rmin * 0.85), self._rmax * 1.08)
        if t > self.ax.max():
            _nice(self.ax, 0, t * 1.25)
            self.ax.setLabelFormat("%.0f")

    def load(self, points: list[tuple[float, float | None, float | None]]):
        self.clear()
        for t, l, r in points:
            self.add(t, l, r)
