"""Catalogue des modèles d'IA installables (bibliothèque Ollama) et compatibilité avec le PC.

Source : https://ollama.com/library (page publique). Une copie est livrée avec le logiciel
(tflops_meter/data/catalogue.json) ; le bouton « Mettre à jour » la remplace par la liste du jour,
enregistrée dans le dossier utilisateur.
"""
from __future__ import annotations

import html
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from . import ai_profile as ap
from .paths import data_dir, resource_dir

BASE = "https://ollama.com"
UA = {"User-Agent": "TFLOPS-Meter/1.0 (catalogue de modeles)"}
USER_FILE = os.path.join(data_dir(), "catalogue.json")
BUNDLED_FILE = os.path.join(resource_dir(), "data", "catalogue.json")

CAPABILITIES = {"tools", "vision", "thinking", "embedding", "audio", "cloud"}
# Modèles MoE dont l'étiquette ne dit pas le nombre de paramètres actifs (milliards)
MOE_ACTIVE_HINTS = {"gpt-oss:20b": 3.6, "gpt-oss:120b": 5.1, "qwen3:30b": 3.3, "qwen3:235b": 22,
                    "qwen3-coder:30b": 3.3, "deepseek-r1:671b": 37, "deepseek-v3:671b": 37,
                    "mixtral:8x7b": 12.9, "mixtral:8x22b": 39, "llama4:16x17b": 17, "llama4:128x17b": 17}


class CatalogError(RuntimeError):
    pass


def _fetch(url: str, timeout=25) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _num_suffix(s: str) -> float | None:
    m = re.fullmatch(r"([\d.]+)\s*([KMB])?", s.strip(), re.I)
    if not m:
        return None
    v = float(m.group(1))
    return v * {"K": 1e3, "M": 1e6, "B": 1e9, None: 1}[m.group(2).upper() if m.group(2) else None]


def parse_size_tag(tag: str) -> tuple[float | None, float | None]:
    """« 8b » → (8, 8) ; « 30b-a3b » → (30, 3) ; « 8x7b » → (56, ~13) ; « e4b » → (4, 4) ; « 335m » → (0.335, …)."""
    t = tag.lower()
    m = re.search(r"(\d+)x([\d.]+)b", t)
    if m:
        n, each = int(m.group(1)), float(m.group(2))
        return n * each, each * 2 * 0.92
    m = re.search(r"(?<![a-z])e?([\d.]+)([bm])", t)
    if not m:
        return None, None
    total = float(m.group(1)) / (1000 if m.group(2) == "m" else 1)
    a = re.search(r"-a([\d.]+)b", t)
    return total, float(a.group(1)) if a else total


# --------------------------------------------------------------------------- analyse des pages

def parse_library(page: str) -> list[dict]:
    out, seen = [], set()
    for block in page.split('href="/library/')[1:]:
        name = block.split('"', 1)[0]
        if not re.fullmatch(r"[a-z0-9._-]+", name) or name in seen:
            continue
        body = block[:6000]
        title = re.search(r'title="([^"]+)"', body)
        if not title or title.group(1) != name:
            continue
        seen.add(name)
        desc = re.search(r"<p[^>]*>(.*?)</p>", body, re.S)
        spans = [html.unescape(s).strip() for s in re.findall(r"<span\s+class=\"[^\"]*\">([^<]+)</span>", body)]
        caps = [s for s in spans if s.lower() in CAPABILITIES]
        sizes = [s for s in spans if re.fullmatch(r"(e?[\d.]+[bm](-a[\d.]+b)?|\d+x[\d.]+b)", s.lower())]
        pulls = re.search(r"<span\s*>([\d.]+[KMB]?)</span>\s*<span class=\"hidden sm:flex\">&nbsp;Pulls", body)
        updated = re.search(r'<span class="flex items-center" title="([^"]+)"', body)
        out.append({
            "name": name,
            "description": html.unescape(re.sub(r"\s+", " ", desc.group(1))).strip() if desc else "",
            "capabilities": caps, "sizes": sizes,
            "pulls": _num_suffix(pulls.group(1)) if pulls else None,
            "updated": updated.group(1) if updated else "",
        })
    return out


# « 9.3GB · 40K context window · Text · » ou, pour les modèles multimodaux, « 6.6GB - 9.5GB · … »
_TAG_RE = re.compile(
    r">([a-z0-9._-]+:[a-z0-9._-]+)</p>\s*</span>\s*<p class=\"flex text-neutral-500\">\s*"
    r"([\d.]+)\s*(GB|MB)(?:\s*-\s*([\d.]+)\s*(GB|MB))?\s*·\s*([\d.]+[KM]?)\s*context window\s*·\s*([^·<]+?)\s*·",
    re.S)


def parse_model_page(page: str) -> list[dict]:
    variants, seen = [], set()
    for tag, size, unit, size2, unit2, ctx, inputs in _TAG_RE.findall(page):
        if tag in seen or "cloud" in tag:
            continue
        seen.add(tag)
        gb = float(size) / (1000 if unit == "MB" else 1)
        if size2:  # fourchette : on garde la borne haute (poids + encodeurs vision/audio)
            gb = max(gb, float(size2) / (1000 if unit2 == "MB" else 1))
        variants.append({"tag": tag, "size_gb": gb, "ctx": int(_num_suffix(ctx) or 0) or None,
                         "inputs": [i.strip() for i in inputs.split(",")]})
    # « latest » est un alias : on le retire s'il double une variante de même taille, sinon on le garde
    # (ex. llama3.1:8b n'existe que sous « latest ») ; entries() retrouve alors sa vraie taille.
    sizes = {round(v["size_gb"], 1) for v in variants if not v["tag"].endswith(":latest")}
    return [v for v in variants if not (v["tag"].endswith(":latest") and round(v["size_gb"], 1) in sizes)]


# --------------------------------------------------------------------------- mise à jour

def update(progress=None, cancel=None) -> dict:
    """Télécharge la bibliothèque Ollama et la page de chaque modèle. Écrit USER_FILE."""
    if progress:
        progress(0, 1, "Liste des modèles…")
    try:
        families = parse_library(_fetch(BASE + "/library?sort=popular"))
    except OSError as e:
        raise CatalogError(f"ollama.com inaccessible : {e}") from e
    if len(families) < 20:
        raise CatalogError("La page de la bibliothèque Ollama a changé de format : mise à jour impossible.")
    done = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch, f"{BASE}/library/{f['name']}"): f for f in families}
        for fut in as_completed(futures):
            if cancel and cancel():
                pool.shutdown(cancel_futures=True)
                raise CatalogError("Mise à jour annulée")
            fam = futures[fut]
            try:
                fam["variants"] = parse_model_page(fut.result())
            except OSError:
                fam["variants"] = []
            done += 1
            if progress:
                progress(done, len(families), fam["name"])
    cat = {"source": BASE + "/library", "fetched": datetime.now().isoformat(timespec="seconds"),
           "models": families}
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(cat, f, ensure_ascii=False)
    return cat


def load() -> dict:
    for path in (USER_FILE, BUNDLED_FILE):
        try:
            with open(path, encoding="utf-8") as f:
                cat = json.load(f)
            if cat.get("models"):
                cat["origin"] = "mise à jour" if path == USER_FILE else "liste intégrée"
                return cat
        except (OSError, ValueError):
            continue
    return {"models": [], "fetched": None, "origin": "aucune"}


# --------------------------------------------------------------------------- compatibilité

USAGES = ["Discussion", "Code", "Vision", "Raisonnement", "Agents / outils", "Embeddings", "OCR"]


def usages(fam: dict, variant: dict | None = None) -> list[str]:
    text = (fam["name"] + " " + fam.get("description", "")).lower()
    caps = set(fam.get("capabilities") or [])
    u = []
    if "embedding" in caps or "embed" in fam["name"]:
        return ["Embeddings"]
    u.append("Discussion")
    if "code" in text or "coder" in text or "programming" in text:
        u.append("Code")
    if "vision" in caps or (variant and "Image" in (variant.get("inputs") or [])) or "-vl" in fam["name"]:
        u.append("Vision")
    if "ocr" in text:
        u.append("OCR")
    if "thinking" in caps or "reasoning" in text or "-r1" in fam["name"]:
        u.append("Raisonnement")
    if "tools" in caps:
        u.append("Agents / outils")
    return u


def _kv_guess(total_b: float, moe: bool) -> int:
    # Ordre de grandeur des architectures récentes (GQA) ; les MoE ont un cache plus petit.
    return int(35_000 * max(total_b, 0.5) ** 0.6 * (0.4 if moe else 1.0))


def entries(cat: dict, hw: ap.Hardware, ctx: int) -> list[dict]:
    """Une entrée par variante locale (les variantes « cloud » sont ignorées)."""
    out = []
    for fam in cat.get("models", []):
        if "cloud" in (fam.get("capabilities") or []) and not fam.get("sizes"):
            continue
        variants = fam.get("variants") or []
        if not variants:  # page non téléchargée : taille déduite de l'étiquette (Q4_K_M ≈ 0,6 Go / milliard)
            for s in fam.get("sizes", []):
                total, _ = parse_size_tag(s)
                if total:
                    variants.append({"tag": f"{fam['name']}:{s}", "size_gb": total * 0.6 + 0.2, "ctx": None,
                                     "inputs": [], "estimated_size": True})
        desc = fam.get("description", "")
        moe_desc = re.search(r"mixture[- ]of[- ]experts|\bMoE\b|active parameters", desc, re.I)
        for v in variants:
            if "cloud" in v["tag"]:
                continue
            size_part = v["tag"].split(":", 1)[1]
            total, active = parse_size_tag(size_part)
            if total is None:  # étiquette sans taille (« latest ») : la taille annoncée la plus proche du fichier
                cands = [(parse_size_tag(s)[0], s) for s in fam.get("sizes", [])]
                cands = [(t, s) for t, s in cands if t]
                if cands:
                    total, s = min(cands, key=lambda c: abs(c[0] * 0.6 - v["size_gb"]))
                    total, active = parse_size_tag(s)
                    v = {**v, "tag": f"{fam['name']}:{s}"}
                else:
                    total = active = v["size_gb"] / 0.6
            hint = MOE_ACTIVE_HINTS.get(v["tag"])
            if hint:
                active = hint
            elif active == total and moe_desc and total >= 16 and "dense" not in desc.lower():
                # (familles mixtes « dense et MoE » : sans indice, on ne devine pas)
                # MoE annoncé sans nombre de paramètres actifs : « 35B-A3B » dans la description, sinon ~12 %
                m = re.search(rf"{total:g}B[- ]A([\d.]+)B", desc, re.I)
                active = float(m.group(1)) if m else total * 0.12
            moe = bool(active and total and active < total * 0.8)
            active_gb = v["size_gb"] * ((active / total) if moe else 1.0)
            use_ctx = min(ctx, v["ctx"]) if v.get("ctx") else ctx
            e = ap.estimate(hw, v["size_gb"], active_gb, _kv_guess(total, moe), use_ctx, active, v.get("ctx"), moe)
            out.append({
                "family": fam["name"], "tag": v["tag"], "description": fam.get("description", ""),
                "usages": usages(fam, v), "size_gb": v["size_gb"], "params_b": total, "active_b": active,
                "moe": moe, "ctx": v.get("ctx"), "pulls": fam.get("pulls") or 0, "updated": fam.get("updated"),
                "estimated_size": v.get("estimated_size", False), "estimate": e,
            })
    return out


def best_per_family(items: list[dict], min_tps: float = 15) -> list[dict]:
    """Pour chaque modèle, la plus grosse variante qui tourne bien (sinon la plus rapide)."""
    by = {}
    for it in items:
        by.setdefault(it["family"], []).append(it)
    out = []
    for fam, lst in by.items():
        good = [i for i in lst if (i["estimate"].gen_tps or 0) >= min_tps and i["estimate"].possible]
        if good:
            out.append(max(good, key=lambda i: i["size_gb"]))
        else:
            ok = [i for i in lst if i["estimate"].possible]
            if ok:
                out.append(max(ok, key=lambda i: i["estimate"].gen_tps or 0))
    return out
