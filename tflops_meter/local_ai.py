"""Outils d'IA locale : détection (Ollama, LM Studio, ComfyUI) et mesure réelle via Ollama."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA
OLLAMA = OLLAMA.replace("0.0.0.0", "127.0.0.1").rstrip("/")
LMSTUDIO = "http://127.0.0.1:1234"
COMFYUI = "http://127.0.0.1:8188"
COMFYUI_URLS = [COMFYUI, "http://127.0.0.1:8000"]


def _get(url, timeout=1.5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _installed(*paths):
    return next((p for p in paths if p and os.path.exists(p)), None)


# --------------------------------------------------------------------------- modèles Ollama

def _arch(info: dict) -> dict:
    """Extrait l'architecture utile au calcul mémoire depuis /api/show (clés « <famille>.xxx »)."""
    def find(suffix):
        for k, v in info.items():
            if k.endswith("." + suffix):
                return v
        return None

    def num(v):
        if isinstance(v, list):  # certains modèles donnent une valeur par couche
            v = max(v) if v else None
        return v
    layers = num(find("block_count"))
    kv_heads = num(find("attention.head_count_kv")) or num(find("attention.head_count"))
    key_len = num(find("attention.key_length"))
    val_len = num(find("attention.value_length"))
    emb = num(find("embedding_length"))
    heads = num(find("attention.head_count"))
    if not key_len and emb and heads:
        key_len = emb // heads
    val_len = val_len or key_len
    kv_per_token = None
    if layers and kv_heads and key_len:
        kv_per_token = layers * kv_heads * (key_len + val_len) * 2  # cache K+V en FP16
    return {
        "layers": layers, "kv_heads": kv_heads, "head_dim": key_len,
        "kv_bytes_per_token": kv_per_token,
        "experts": num(find("expert_count")), "experts_used": num(find("expert_used_count")),
        "sliding_window": num(find("attention.sliding_window")),
    }


def ollama_models() -> list[dict]:
    tags = _get(OLLAMA + "/api/tags", timeout=3)
    out = []
    for m in tags.get("models", []):
        d = m.get("details", {}) or {}
        entry = {
            "name": m["name"], "size_bytes": m.get("size", 0),
            "params": d.get("parameter_size", ""), "quant": d.get("quantization_level", ""),
            "family": d.get("family", ""), "max_ctx": d.get("context_length"),
            "capabilities": m.get("capabilities") or [],
        }
        try:
            show = _post(OLLAMA + "/api/show", {"model": m["name"]}, timeout=10)
            entry.update(_arch(show.get("model_info") or {}))
            caps = show.get("capabilities")
            if caps:
                entry["capabilities"] = caps
            if not entry["max_ctx"]:
                for k, v in (show.get("model_info") or {}).items():
                    if k.endswith(".context_length"):
                        entry["max_ctx"] = v
        except (urllib.error.URLError, OSError, ValueError):
            pass
        entry["embedding_only"] = "embedding" in entry["capabilities"] and "completion" not in entry["capabilities"]
        out.append(entry)
    return out


def detect() -> dict:
    """État des outils d'IA locale. Ne lève jamais d'exception."""
    local = os.environ.get("LOCALAPPDATA", "")
    home = os.path.expanduser("~")
    res = {"ollama": {"running": False}, "lmstudio": {"running": False}, "comfyui": {"running": False}}

    o = res["ollama"]
    o["installed"] = _installed(os.path.join(local, "Programs", "Ollama", "ollama.exe"),
                                os.path.join(home, ".ollama"))
    try:
        o["version"] = _get(OLLAMA + "/api/version")["version"]
        o["running"] = True
        o["models"] = ollama_models()
        o["loaded"] = _get(OLLAMA + "/api/ps").get("models", [])
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        pass

    lm = res["lmstudio"]
    lm["installed"] = _installed(os.path.join(local, "Programs", "LM Studio", "LM Studio.exe"),
                                 os.path.join(home, ".lmstudio"), os.path.join(home, ".cache", "lm-studio"))
    try:
        lm["models"] = [m["id"] for m in _get(LMSTUDIO + "/v1/models", timeout=1.0).get("data", [])]
        lm["running"] = True
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        pass

    c = res["comfyui"]
    for url in COMFYUI_URLS:  # ComfyUI classique (8188) ou ComfyUI Desktop (8000)
        try:
            st = _get(url + "/system_stats", timeout=1.0)
            if "system" not in st:
                continue
            c["running"], c["url"] = True, url
            c["version"] = (st.get("system") or {}).get("comfyui_version")
            devs = st.get("devices") or []
            if devs:
                c["vram_free_gb"] = devs[0].get("vram_free", 0) / 1e9
                c["vram_total_gb"] = devs[0].get("vram_total", 0) / 1e9
            break
        except (urllib.error.URLError, OSError, ValueError, AttributeError):
            continue
    from .image_models import MODEL_DIRS, comfy_base_paths, local_model_files
    files = []
    if c["running"]:
        for d in MODEL_DIRS:
            try:
                files += [f for f in _get(f"{c['url']}/models/{d}", timeout=2.0) if isinstance(f, str)]
            except (urllib.error.URLError, OSError, ValueError, TypeError):
                pass
    try:
        files += local_model_files()
        c["installed"] = (comfy_base_paths() or [None])[0]
    except OSError:
        pass
    c["model_files"] = sorted(set(files))
    return res


# --------------------------------------------------------------------------- mesure réelle

_PROMPT_BASE = ("Voici des notes techniques à résumer. Les processeurs modernes exécutent des instructions "
                "vectorielles, les cartes graphiques possèdent des milliers d'unités de calcul, et la mémoire "
                "vidéo offre une bande passante bien supérieure à la mémoire système. ")


def ollama_benchmark(model: str, num_ctx: int = 8192, predict: int = 256) -> dict:
    """Charge le modèle, mesure la génération (tok/s), le traitement du prompt et la part en VRAM."""
    prompt = _PROMPT_BASE * 12 + ("\n\nRédige maintenant une explication détaillée, en plusieurs paragraphes, "
                                  "de la différence entre bande passante mémoire et puissance de calcul.")
    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": "2m",
               "options": {"num_ctx": num_ctx, "num_predict": predict, "temperature": 0, "seed": 1}}
    t0 = time.perf_counter()
    try:
        r = _post(OLLAMA + "/api/generate", {**payload, "think": False})
    except urllib.error.HTTPError as e:  # anciens Ollama / modèles sans option « think »
        if e.code != 400:
            raise
        r = _post(OLLAMA + "/api/generate", payload)
    wall = time.perf_counter() - t0
    gen = r.get("eval_count", 0) / (r.get("eval_duration", 1) / 1e9) if r.get("eval_duration") else None
    pp = (r.get("prompt_eval_count", 0) / (r.get("prompt_eval_duration", 1) / 1e9)
          if r.get("prompt_eval_duration") else None)
    gpu_pct, size_gb = None, None
    try:
        for m in _get(OLLAMA + "/api/ps").get("models", []):
            if m.get("name") == model or m.get("model") == model:
                size = m.get("size") or 0
                size_gb = size / 1e9
                gpu_pct = 100 * (m.get("size_vram") or 0) / size if size else None
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return {
        "model": model, "gen_tps": gen, "prompt_tps": pp, "load_s": r.get("load_duration", 0) / 1e9,
        "tokens": r.get("eval_count"), "prompt_tokens": r.get("prompt_eval_count"), "wall_s": wall,
        "gpu_pct": gpu_pct, "loaded_gb": size_gb, "num_ctx": num_ctx,
    }


def ollama_unload(model: str):
    try:
        _post(OLLAMA + "/api/generate", {"model": model, "keep_alive": 0}, timeout=30)
    except (urllib.error.URLError, OSError, ValueError):
        pass
