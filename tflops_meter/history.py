"""Historique des mesures (SQLite) et export JSON / CSV."""
from __future__ import annotations

import csv
import json
import os
import sqlite3
import threading
from datetime import datetime

from .model import Measure
from .paths import data_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT NOT NULL,
    kind TEXT NOT NULL,              -- 'analyse' ou 'stabilite'
    preset TEXT,
    label TEXT,
    system_json TEXT,
    summary_json TEXT
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    category TEXT, device TEXT, test TEXT,
    value REAL, unit TEXT, theoretical REAL,
    details TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS samples (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    t REAL, perf REAL, temp_c REAL, clock_mhz REAL, power_w REAL, throttle TEXT
);
CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_samples_run ON samples(run_id);
"""


class History:
    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(data_dir(), "historique.db")
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def save_run(self, kind: str, preset: str, system: dict, measures: list[Measure],
                 summary: dict | None = None, samples: list[dict] | None = None, label: str = "") -> int:
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO runs(created, kind, preset, label, system_json, summary_json) VALUES (?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), kind, preset, label,
                 json.dumps(system, ensure_ascii=False), json.dumps(summary or {}, ensure_ascii=False)))
            run_id = cur.lastrowid
            self.db.executemany(
                "INSERT INTO results(run_id, category, device, test, value, unit, theoretical, details, error) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                [(run_id, m.category, m.device, m.test, m.value, m.unit, m.theoretical, m.details, m.error)
                 for m in measures])
            if samples:
                self.db.executemany(
                    "INSERT INTO samples(run_id, t, perf, temp_c, clock_mhz, power_w, throttle) VALUES (?,?,?,?,?,?,?)",
                    [(run_id, s.get("t"), s.get("perf"), s.get("temp_c"), s.get("clock_mhz"), s.get("power_w"),
                      ", ".join(s.get("throttle") or [])) for s in samples])
            self.db.commit()
            return run_id

    def runs(self, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self.db.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["system"] = json.loads(d.pop("system_json") or "{}")
                d["summary"] = json.loads(d.pop("summary_json") or "{}")
                out.append(d)
            return out

    def run(self, run_id: int) -> dict | None:
        with self._lock:
            r = self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if r is None:
                return None
            d = dict(r)
            d["system"] = json.loads(d.pop("system_json") or "{}")
            d["summary"] = json.loads(d.pop("summary_json") or "{}")
            d["results"] = [dict(x) for x in self.db.execute(
                "SELECT category, device, test, value, unit, theoretical, details, error FROM results "
                "WHERE run_id=? ORDER BY id", (run_id,))]
            d["samples"] = [dict(x) for x in self.db.execute(
                "SELECT t, perf, temp_c, clock_mhz, power_w, throttle FROM samples WHERE run_id=? ORDER BY t",
                (run_id,))]
            return d

    def delete(self, run_id: int):
        with self._lock:
            self.db.execute("DELETE FROM runs WHERE id=?", (run_id,))
            self.db.commit()

    def rename(self, run_id: int, label: str):
        with self._lock:
            self.db.execute("UPDATE runs SET label=? WHERE id=?", (label, run_id))
            self.db.commit()


# --------------------------------------------------------------------------- export

def export_json(runs: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"application": "TFLOPS Meter", "exported": datetime.now().isoformat(timespec="seconds"),
                   "runs": runs}, f, ensure_ascii=False, indent=2)


def export_csv(runs: list[dict], path: str):
    """Une ligne par mesure ; les échantillons de stabilité vont dans un second fichier *_stabilite.csv."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["run_id", "date", "type", "categorie", "materiel", "test", "valeur", "unite",
                    "theorique", "efficacite_%", "details", "erreur"])
        for r in runs:
            for m in r.get("results", []):
                eff = (m["value"] / m["theoretical"] * 100) if m.get("value") and m.get("theoretical") else None
                w.writerow([r.get("id"), r.get("created"), r.get("kind"), m["category"], m["device"], m["test"],
                            _num(m["value"]), m["unit"], _num(m["theoretical"]), _num(eff),
                            m.get("details") or "", m.get("error") or ""])
    samples = [(r, s) for r in runs for s in r.get("samples", [])]
    if samples:
        base, ext = os.path.splitext(path)
        with open(f"{base}_stabilite{ext}", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["run_id", "t_s", "performance", "temp_c", "frequence_mhz", "puissance_w", "bridage"])
            for r, s in samples:
                w.writerow([r.get("id"), _num(s["t"]), _num(s["perf"]), _num(s["temp_c"]),
                            _num(s["clock_mhz"]), _num(s["power_w"]), s.get("throttle") or ""])


def _num(v):
    # virgule décimale : le CSV s'ouvre directement dans Excel en français
    return "" if v is None else f"{v:.4f}".replace(".", ",")
