"""Génération d'images et de vidéos (modèles de diffusion) : catalogue Hugging Face et estimations pour ce PC.

Estimation (ComfyUI, Forge, SwarmUI fonctionnent ainsi) :
- mémoire : poids du modèle de diffusion + activations. L'encodeur de texte et le VAE passent à tour
  de rôle : ComfyUI les décharge en RAM pendant le débruitage ;
- vitesse : une image = nombre d'étapes × (calcul d'une étape ÷ puissance FP16 mesurée). Le guidage
  (CFG) double le calcul. Si le modèle déborde de la VRAM, la partie en RAM transite par le PCIe à
  chaque étape ;
- calcul d'une étape ≈ k × milliards de paramètres (TFLOP), k dépendant de l'architecture et du
  nombre de « tokens » d'image (résolution, nombre d'images d'une vidéo).
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

from . import ai_profile as ap
from .paths import data_dir, resource_dir

HF = "https://huggingface.co"
UA = {"User-Agent": "TFLOPS-Meter/1.0 (catalogue de modeles image)"}
USER_FILE = os.path.join(data_dir(), "catalogue_images.json")
BUNDLED_FILE = os.path.join(resource_dir(), "data", "catalogue_images.json")

PIPELINES = {"text-to-image": "image", "image-to-image": "image", "image-text-to-image": "image",
             "text-to-video": "video", "image-to-video": "video"}

# Calibrage : SDXL 1024² 30 étapes ≈ 4,5 s et Flux.1 dev 20 étapes ≈ 18 s sur RTX 4090 (GEMM FP16 ≈ 150 TFLOPS)
GEN_EFF = 0.55        # relatif à la GEMM FP16 cuBLAS mesurée
OTHER_GPU_EFF = 0.5   # AMD / Intel sous Windows (DirectML, ZLUDA, ROCm) : PyTorch nettement moins efficace
CPU_EFF = 0.35        # relatif aux GFLOPS FP32 multicœur mesurés
PCIE_GBS = 20.0       # transfert RAM → VRAM effectif quand le modèle déborde
OVERHEAD_GB = 0.5     # contexte CUDA + PyTorch

# k : TFLOP par passe et par milliard de paramètres à la résolution de référence
PROFILES = {
    "sd15": dict(label="Stable Diffusion 1.5", ref="512×512", k=1.5, steps=25, cfg=True, act=0.6,
                 te="CLIP-L", te_gb=0.25, vae_gb=0.17, checkpoint=True),
    "sdxl": dict(label="SDXL", ref="1024×1024", k=2.5, steps=25, cfg=True, act=1.5,
                 te="CLIP-L + CLIP-G", te_gb=1.6, vae_gb=0.17, checkpoint=True),
    "sd3": dict(label="Stable Diffusion 3.x", ref="1024×1024", k=3.7, steps=28, cfg=True, act=1.5,
                te="CLIP-L, CLIP-G et T5-XXL en FP8", te_gb=6.4, vae_gb=0.17),
    "flux1": dict(label="Flux.1", ref="1024×1024", k=6.2, steps=20, cfg=False, act=1.8,
                  te="T5-XXL en FP8 + CLIP-L", te_gb=5.2, vae_gb=0.34),
    "flux2": dict(label="Flux.2", ref="1024×1024", k=6.1, steps=28, cfg=False, act=2.0,
                  te="encodeur de texte LLM en FP8", te_gb=None, vae_gb=0.34),
    "qwen": dict(label="Qwen-Image", ref="1024×1024", k=3.4, steps=25, cfg=True, act=2.0,
                 te="Qwen2.5-VL 7B en FP8", te_gb=9.4, vae_gb=0.25),
    "zimage": dict(label="Z-Image", ref="1024×1024", k=9.0, steps=8, cfg=False, act=1.5,
                   te="Qwen3 4B en FP8", te_gb=4.0, vae_gb=0.34),
    "dit": dict(label="Transformer de diffusion", ref="1024×1024", k=6.0, steps=28, cfg=True, act=2.0,
                te="encodeur de texte (taille estimée)", te_gb=5.0, vae_gb=0.3),
    "wan": dict(label="Wan", ref="clip de 5 s en 480p", k=130, steps=20, cfg=True, act=3.5,
                te="UMT5-XXL en FP8", te_gb=6.7, vae_gb=0.25),
    "ltx": dict(label="LTX", ref="clip de 5 s en 720p", k=15, steps=20, cfg=True, act=3.0,
                te="encodeur de texte (taille estimée)", te_gb=12.0, vae_gb=1.0),
    "video": dict(label="Vidéo (profil générique)", ref="clip de 5 s en 480p", k=100, steps=20, cfg=True, act=3.5,
                  te="encodeur de texte (taille estimée)", te_gb=6.7, vae_gb=0.3),
}

# Mots du nom qui signalent un modèle distillé en peu d'étapes, sans guidage
FEW_STEPS = [("schnell", 4), ("klein", 4), ("lightning", 8), ("turbo", 8), ("distill", 8), ("hyper", 8),
             ("lcm", 6), ("rapid", 6), ("fast", 8)]

USAGES = ["Texte → image", "Édition d'image", "Texte → vidéo", "Image → vidéo", "Anime / illustration",
          "Photo réaliste"]

# Descriptions des modèles les plus connus (clé normalisée → texte)
DESCRIPTIONS = {
    "stablediffusionv15": "Le modèle historique : très léger, des milliers de variantes sur Civitai. Qualité datée.",
    "stablediffusionxlbase10": "Référence 1024×1024, base de très nombreuses variantes (Pony, Illustrious, "
                               "Juggernaut, RealVis…).",
    "sdxlturbo": "SDXL distillé en 1 à 4 étapes : quasi instantané, qualité moindre.",
    "flux1dev": "Très bonne qualité et texte lisible dans l'image. Licence non commerciale.",
    "flux1schnell": "Version rapide de Flux.1 (4 étapes), licence Apache 2.0.",
    "flux1kontextdev": "Édition d'image par instruction écrite, basé sur Flux.1.",
    "flux2dev": "Flux de deuxième génération (32 B) : très haute qualité, mais lourd, avec un encodeur de texte "
                "de 24 Go.",
    "qwenimage": "Très bon rendu du texte dans l'image, licence Apache 2.0.",
    "zimageturbo": "Rapide (8 étapes) et photoréaliste, licence Apache 2.0.",
    "wan21t2v13b": "Petit modèle vidéo, accessible avec 8 Go de VRAM.",
    "wan22ti2v5b": "Vidéo 720p à partir d'un texte ou d'une image, bon compromis taille / qualité.",
    "wan22t2va14b": "Vidéo de haute qualité : deux modèles de 14 B chargés à tour de rôle. Les LoRA "
                    "« lightx2v » réduisent à 4 étapes.",
    "wan22i2va14b": "Image → vidéo de haute qualité : deux modèles de 14 B chargés à tour de rôle. Les LoRA "
                    "« lightx2v » réduisent à 4 étapes.",
}
ALIASES = {  # noms de fichiers usuels dans ComfyUI
    "stablediffusionv15": ["v15prunedemaonly", "sd15", "v15pruned"],
    "stablediffusionxlbase10": ["sdxlbase10", "sdxlbase"],
    "flux1dev": ["flux1dev"],
    "flux1schnell": ["flux1schnell"],
    "zimageturbo": ["zimageturbo", "zimage"],
    "wan22ti2v5b": ["wan22ti2v5b"],
    "wan22t2va14b": ["wan22t2vhighnoise14b", "wan22t2vlownoise14b"],
    "wan22i2va14b": ["wan22i2vhighnoise14b", "wan22i2vlownoise14b"],
}
CURATED = ["stable-diffusion-v1-5/stable-diffusion-v1-5", "stabilityai/stable-diffusion-xl-base-1.0",
           "black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-Kontext-dev",
           "black-forest-labs/FLUX.2-dev", "Qwen/Qwen-Image", "Tongyi-MAI/Z-Image-Turbo",
           "stabilityai/stable-diffusion-3.5-large", "stabilityai/stable-diffusion-3.5-medium",
           "Wan-AI/Wan2.1-T2V-1.3B", "Wan-AI/Wan2.2-TI2V-5B", "Wan-AI/Wan2.2-T2V-A14B", "Wan-AI/Wan2.2-I2V-A14B"]

REPACKAGERS = {"comfy-org", "unsloth", "quantstack", "city96", "calcuis", "bullerwins", "kijai", "comfyanonymous",
               "gguf-org", "leejet", "abiray", "second-state", "jayn7", "nunchaku-tech", "lightx2v", "comfyicu",
               "ml-resources", "t8star", "chriscoletech"}
REPACK_LIBS = {"gguf", "ggml", "diffusion-single-file", "mflux", "audio.cpp"}
REPACK_WORDS = ("gguf", "fp8", "nf4", "fp4", "int8", "int4", "svdq", "nunchaku", "comfy", "bnb", "4bit", "8bit",
                "6bit", "mlx", "onnx", "openvino", "awq", "gptq")
NOISE_WORDS = ("lora", "controlnet", "ipadapter", "ip-adapter", "vae", "upscal", "decoder", "encoder", "adapter",
               "lycoris", "embedding", "inpaint-mask", "face-swap", "faceswap", "aio")  # AIO : taille faussée
STRIP_WORDS = ("diffusers", "gguf", "fp8", "bf16", "fp16", "nf4", "comfyui", "comfy", "scaled", "e4m3fn",
               "quantized", "repackaged", "singlefile", "safetensors")


class ImageCatalogError(Exception):
    pass


def norm(name: str) -> str:
    s = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    for w in STRIP_WORDS:
        s = s.replace(w, "")
    return s


def _key(repo_id: str) -> str:
    return norm(repo_id.split("/")[-1])


def pretty_name(repo_id: str) -> str:
    n = repo_id.split("/")[-1]
    return re.sub(r"[-_](diffusers|Diffusers)$", "", n)


# --------------------------------------------------------------------------- Hugging Face

_EXPAND = ["safetensors", "downloads", "likes", "tags", "library_name", "lastModified", "gated", "pipeline_tag"]


def _get_json(url: str, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _list(pipeline: str, limit=300) -> list[dict]:
    q = [("pipeline_tag", pipeline), ("sort", "downloads"), ("limit", str(limit))]
    q += [("expand[]", e) for e in _EXPAND]
    return _get_json(f"{HF}/api/models?{urllib.parse.urlencode(q)}")


def _details(repo_id: str) -> dict | None:
    q = urllib.parse.urlencode([("expand[]", e) for e in _EXPAND])
    try:
        return _get_json(f"{HF}/api/models/{repo_id}?{q}", timeout=20)
    except (OSError, ValueError):
        return None


_TE_WORDS = ("text_encoder", "textencoder", "clip", "t5", "umt5", "vae", "qwen_2.5_vl", "qwen2.5", "mistral",
             "gemma", "llama", "lora", "tokenizer", "qwen3_")


def _params_from_files(repo_id: str) -> float | None:
    """Repli quand l'API ne donne pas le nombre de paramètres : taille du plus gros fichier du modèle de diffusion."""
    try:
        files = _get_json(f"{HF}/api/models/{repo_id}/tree/main?recursive=true", timeout=20)
    except (OSError, ValueError):
        return None
    best = None
    for f in files:
        path = (f.get("path") or "").lower()
        if f.get("type") != "file" or not path.endswith((".safetensors", ".gguf")):
            continue
        if any(w in path for w in _TE_WORDS) and "diffusion_model" not in path:
            continue
        size = f.get("size") or 0
        if "fp8" in path or "q8" in path:
            bpp = 1.05
        elif "fp4" in path or "nf4" in path or re.search(r"q4|iq4", path):
            bpp = 0.6
        elif re.search(r"q5", path):
            bpp = 0.7
        elif re.search(r"q[236]", path):
            bpp = 0.5
        else:
            bpp = 2.0
        p = size / bpp / 1e9
        if best is None or p > best:
            best = p
    return round(best, 2) if best and best > 0.2 else None


def _base_target(tags: list[str]) -> str | None:
    for kind in ("quantized", "finetune", "merge"):
        for t in tags:
            if t.startswith(f"base_model:{kind}:"):
                return t.split(":", 2)[2]
    return None


def _is_noise(r: dict) -> bool:
    name = r["id"].split("/")[-1].lower()
    tags = r.get("tags") or []
    if any(t.startswith("base_model:adapter:") for t in tags) or "lora" in tags:
        return True
    return any(w in name for w in NOISE_WORDS)


def _is_repack(r: dict) -> bool:
    owner, name = r["id"].lower().split("/", 1) if "/" in r["id"] else ("", r["id"].lower())
    return owner in REPACKAGERS or (r.get("library_name") or "") in REPACK_LIBS or any(w in name for w in REPACK_WORDS)


def _family_from(r: dict, downloads: int = 0) -> dict:
    st = r.get("safetensors") or {}
    tags = r.get("tags") or []
    lic = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)
    arch = next((t.split(":", 1)[1] for t in tags if t.startswith("diffusers:")), None)
    base = _base_target(tags)
    return {"id": r["id"], "name": pretty_name(r["id"]), "pipeline": r.get("pipeline_tag"),
            "params_b": round(st["total"] / 1e9, 2) if st.get("total") else None,
            "downloads": downloads or r.get("downloads") or 0, "likes": r.get("likes") or 0,
            "updated": (r.get("lastModified") or "")[:10], "gated": bool(r.get("gated")), "license": lic,
            "arch": arch, "base": base if base and base != r["id"] else None, "gguf_repo": None, "comfy_repo": None}


def update(progress=None, cancel=None, max_families=70) -> dict:
    """Construit le catalogue depuis l'API Hugging Face (modèles les plus téléchargés sur 30 jours)."""
    def step(done, total, txt):
        if cancel and cancel():
            raise ImageCatalogError("Mise à jour annulée")
        if progress:
            progress(done, total, txt)

    step(0, 10, "Listes Hugging Face…")
    repos: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(_list, p): p for p in PIPELINES}
        for fut in as_completed(futs):
            try:
                for r in fut.result():
                    r.setdefault("pipeline_tag", futs[fut])
                    if r["id"] not in repos:
                        repos[r["id"]] = r
            except (OSError, ValueError) as e:
                raise ImageCatalogError(f"huggingface.co inaccessible : {e}") from e
    if len(repos) < 30:
        raise ImageCatalogError("Réponse inattendue de Hugging Face : mise à jour impossible.")
    step(1, 10, f"{len(repos)} dépôts analysés…")

    fams: dict[str, dict] = {}  # clé normalisée → famille
    repacks = []
    for r in sorted(repos.values(), key=lambda x: -(x.get("downloads") or 0)):
        if _is_noise(r):
            continue
        if _is_repack(r):
            repacks.append(r)
            continue
        k = _key(r["id"])
        if not k:
            continue
        if k in fams:  # « X » et « X-Diffusers » : un seul modèle
            f = fams[k]
            f["downloads"] += r.get("downloads") or 0
            if not f["params_b"] and (r.get("safetensors") or {}).get("total"):
                f["params_b"] = round(r["safetensors"]["total"] / 1e9, 2)
            continue
        fams[k] = _family_from(r)

    def attach(r, target_key, count=True):
        f = fams[target_key]
        if count:
            f["downloads"] += r.get("downloads") or 0
        lib = r.get("library_name") or ""
        owner = r["id"].split("/")[0].lower()
        if lib in ("gguf", "ggml") or "gguf" in r["id"].lower():
            if not f["gguf_repo"]:
                f["gguf_repo"] = r["id"]
        elif owner == "comfy-org" or lib == "diffusion-single-file":
            preferred = owner in ("comfy-org", f["id"].split("/")[0].lower())
            if not f["comfy_repo"] or (preferred and not f.get("comfy_preferred")):
                f["comfy_repo"], f["comfy_preferred"] = r["id"], preferred

    for r in repacks:
        k = _key(r["id"])
        if k in fams:
            attach(r, k)
            continue
        near = [fk for fk in fams if len(k) >= 5 and (fk.startswith(k) or k.startswith(fk))]
        if len(near) == 1:
            attach(r, near[0])
            continue
        base = _base_target(r.get("tags") or [])
        if base and len(near) > 1:  # dépôt qui regroupe plusieurs modèles : lien seulement, sans compter
            if _key(base) in fams:
                attach(r, _key(base), count=False)
            continue
        if base:
            bk = _key(base)
            if bk not in fams:
                fams[bk] = {"id": base, "name": pretty_name(base), "pipeline": r.get("pipeline_tag"),
                            "params_b": None, "downloads": 0, "likes": 0, "updated": "", "gated": False,
                            "license": None, "arch": None, "base": None, "gguf_repo": None, "comfy_repo": None,
                            "placeholder": True}
            attach(r, bk)

    for cid in CURATED:
        k = _key(cid)
        if k not in fams:
            fams[k] = {"id": cid, "name": pretty_name(cid), "pipeline": None, "params_b": None, "downloads": 0,
                       "likes": 0, "updated": "", "gated": False, "license": None, "arch": None, "base": None,
                       "gguf_repo": None, "comfy_repo": None, "placeholder": True}
    keep_keys = {_key(c) for c in CURATED}
    ranked = sorted(fams.items(), key=lambda kv: -kv[1]["downloads"])
    chosen = [kv for i, kv in enumerate(ranked) if (i < max_families and kv[1]["downloads"] >= 5000)
              or kv[0] in keep_keys]

    todo = [f for _, f in chosen if f.get("placeholder") or not f["params_b"]]
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_details, f["id"]): f for f in todo}
        for fut in as_completed(futs):
            f = futs[fut]
            d = fut.result()
            if d:
                g = _family_from(d)
                if g.get("pipeline") and (f.get("placeholder") or not f.get("pipeline")):
                    f["pipeline"] = g["pipeline"]  # un LLM cité à tort comme modèle de base est écarté plus bas
                for key in ("license", "arch", "updated", "gated", "likes"):
                    if g.get(key) and not f.get(key):
                        f[key] = g[key]
                f["params_b"] = f["params_b"] or g["params_b"]
                if f.get("placeholder"):
                    f["downloads"] += d.get("downloads") or 0
                    f.pop("placeholder", None)
                    f["base"] = g["base"]
            else:
                f["dead"] = True
            done += 1
            step(1 + int(done / max(len(todo), 1) * 6), 10, f["name"])

    missing = [f for _, f in chosen if not f.get("dead") and not f["params_b"]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_params_from_files, f["comfy_repo"] or f["gguf_repo"] or f["id"]): f for f in missing}
        for fut in as_completed(futs):
            futs[fut]["params_b"] = fut.result()
            step(8, 10, futs[fut]["name"])

    out = [f for _, f in chosen if not f.get("dead") and f["params_b"] and f.get("pipeline") in PIPELINES]
    for f in out:
        f.pop("placeholder", None)
        f.pop("comfy_preferred", None)
    out.sort(key=lambda f: -f["downloads"])
    cat = {"source": HF + "/models", "fetched": datetime.now().isoformat(timespec="seconds"), "families": out}
    os.makedirs(os.path.dirname(USER_FILE), exist_ok=True)
    with open(USER_FILE, "w", encoding="utf-8") as fh:
        json.dump(cat, fh, ensure_ascii=False, indent=0)
    step(10, 10, "terminé")
    return cat


def load() -> dict:
    for path in (USER_FILE, BUNDLED_FILE):
        try:
            with open(path, encoding="utf-8") as f:
                cat = json.load(f)
            if cat.get("families"):
                cat["origin"] = "mise à jour" if path == USER_FILE else "liste intégrée"
                return cat
        except (OSError, ValueError):
            continue
    return {"families": [], "fetched": None, "origin": "aucune"}


# --------------------------------------------------------------------------- profils

def profile_for(fam: dict) -> dict:
    """Profil de calcul d'un modèle, déduit de son nom, de son architecture et de sa taille."""
    n = _key(fam["id"])
    arch = (fam.get("arch") or "").lower()
    p = fam.get("params_b") or 0
    video = PIPELINES.get(fam.get("pipeline") or "") == "video"
    if video:
        key = "ltx" if "ltx" in n else "wan" if "wan" in n else "video"
    elif "stablediffusionxl" in arch or "sdxl" in n or ("xl" in n and 2.3 <= p <= 4) or 2.5 <= p <= 2.6:
        key = "sdxl"
    elif arch == "stablediffusionpipeline" or 0.8 <= p <= 0.9:
        key = "sd15"
    elif "stablediffusion3" in arch or "stablediffusion3" in n or "sd3" in n:
        key = "sd3"
    elif re.search(r"flux[23]", n):
        key = "flux2"
    elif "flux" in n:
        key = "flux1"
    elif "qwenimage" in n:
        key = "qwen"
    elif "zimage" in n:
        key = "zimage"
    else:
        key = "dit"
    prof = dict(PROFILES[key], key=key, video=video)
    # Les dépôts SD 1.5 / SDXL comptent parfois l'encodeur de texte : on prend la taille du réseau de débruitage
    prof["params"] = {"sd15": 0.86, "sdxl": 2.57}.get(key, p)
    if key == "flux2":  # encodeur de texte : Mistral 24B pour la version dev, plus petit pour les variantes légères
        prof["te_gb"] = 24.0 if p > 20 else max(4.0, round(p * 0.9, 1))
    base_word = "base" in n
    for word, steps in FEW_STEPS:
        if word in n and not base_word:
            prof["steps"], prof["cfg"] = steps, False
            prof["distilled"] = word
            break
    if base_word and key == "flux2":
        prof["cfg"] = True
    prof["edit"] = "edit" in n or "kontext" in n
    if prof["edit"] and not video:
        prof["k"] *= 2.0  # l'image source double le nombre de tokens
    if "a14b" in n:  # Wan 2.2 A14B : deux experts de 14 B chargés l'un après l'autre
        prof["experts"] = 2
    return prof


def usages(fam: dict, prof: dict) -> list[str]:
    n = (fam["id"] + " " + (fam.get("base") or "")).lower()
    pipe = fam.get("pipeline") or ""
    out = []
    if prof["video"]:
        if pipe == "text-to-video" or "t2v" in n or "ti2v" in n:
            out.append("Texte → vidéo")
        if pipe == "image-to-video" or "i2v" in n:
            out.append("Image → vidéo")
        if not out:
            out.append("Texte → vidéo")
    else:
        if not prof["edit"]:
            out.append("Texte → image")
        if prof["edit"] or pipe in ("image-to-image", "image-text-to-image"):
            out.append("Édition d'image")
    if any(w in n for w in ("anime", "pony", "illustrious", "noob", "animagine", "manga", "waifu")):
        out.append("Anime / illustration")
    if any(w in n for w in ("realvis", "juggernaut", "krea", "photo", "realistic", "zimage", "z-image")):
        out.append("Photo réaliste")
    return out


def describe(fam: dict, prof: dict) -> str:
    k = _key(fam["id"])
    if k in DESCRIPTIONS:
        return DESCRIPTIONS[k]
    owner = fam["id"].split("/")[0]
    if prof["key"] in ("dit", "video"):
        kind = "Modèle de vidéo" if prof["video"] else "Modèle de diffusion"
    else:
        kind = f"Famille {prof['label']}"
    txt = f"{kind}, publié par {owner}"
    if fam.get("base"):
        txt += f", dérivé de {fam['base'].split('/')[-1]}"
    if prof.get("distilled"):
        txt += f" ; version rapide ({prof['steps']} étapes)"
    return txt + "."


def formats(prof: dict, p: float) -> list[tuple[str, float, float]]:
    """(format, taille des poids du modèle de diffusion en Go, facteur de lenteur)."""
    if prof.get("checkpoint"):
        return [("FP16", 2.0 * p + prof["te_gb"] + prof["vae_gb"], 1.0)]
    out = [("BF16", 2.0 * p, 1.0), ("FP8", 1.02 * p, 1.0)]
    if p >= 3:
        out += [("GGUF Q8_0", 1.07 * p, 1.1), ("GGUF Q5_K_M", 0.71 * p, 1.25), ("GGUF Q4_K_M", 0.61 * p, 1.25)]
    return out


# --------------------------------------------------------------------------- estimation

def is_nvidia(hw: ap.Hardware) -> bool:
    g = (hw.gpu or "").lower()
    return bool(hw.cc) or any(w in g for w in ("nvidia", "geforce", "rtx", "quadro", "tesla"))


def gpu_tflops(hw: ap.Hardware) -> tuple[float | None, bool]:
    """Puissance FP16 utile pour la diffusion, et vrai si elle vient d'une mesure."""
    v = hw.fp16_tflops or hw.bf16_tflops
    if v:
        return v, True
    if hw.extras.get("fma_fp16"):
        return hw.extras["fma_fp16"] * 0.6, True
    if hw.fp32_tflops:
        return hw.fp32_tflops * 2, False
    if hw.extras.get("fma_fp32"):
        return hw.extras["fma_fp32"] * 1.2, True
    if hw.vram_bw:
        return hw.vram_bw * 0.11, False  # ordre de grandeur : à remplacer par une mesure
    return None, False


@dataclass
class ImgEstimate:
    weights_gb: float
    vram_need_gb: float
    gpu_frac: float
    step_s: float | None
    total_s: float | None
    steps: int
    possible: bool
    fits_now: bool
    video: bool
    cpu: bool
    measured: bool

    @property
    def placement(self):
        if not self.possible:
            return "trop gros"
        if self.cpu:
            return "CPU"
        if self.gpu_frac >= 0.999:
            return "100 % GPU"
        return f"mixte ({self.gpu_frac * 100:.0f} % GPU)"

    @property
    def level(self) -> int:
        """0 très lent … 4 très rapide ; -1 impossible."""
        t = self.total_s
        if not self.possible or t is None:
            return -1
        limits = (120, 600, 1800, 5400) if self.video else (10, 30, 90, 300)
        return 4 - sum(t > x for x in limits)

    @property
    def verdict(self):
        if self.level < 0:
            return "impossible"
        if self.video:
            return ["très lent", "lent", "utilisable", "rapide", "très rapide"][self.level]
        return ["très lent", "lent", "utilisable", "rapide", "très rapide"][self.level]

    @property
    def runs_well(self) -> bool:
        return self.level >= 2


def duration(s: float | None) -> str:
    if s is None:
        return "—"
    if s < 60:
        return f"{s:.0f} s" if s >= 10 else f"{ap.fr(s, 1)} s"
    if s < 3600:
        m = s / 60
        return f"{ap.fr(m, 1)} min" if m < 10 else f"{m:.0f} min"
    return f"{ap.fr(s / 3600, 1)} h"


def estimate(hw: ap.Hardware, prof: dict, p: float, weights_gb: float, slow: float) -> ImgEstimate:
    act = prof["act"]
    usable = hw.usable_gb
    flop_step = prof["k"] * p * (2 if prof["cfg"] else 1)  # TFLOP
    te_gb = prof["te_gb"] or 5.0
    experts = prof.get("experts", 1)
    fixed = 20.0 if prof["video"] else 1.5  # encodage du texte + décodage VAE
    steps = prof["steps"]
    tf, measured = gpu_tflops(hw)
    room = usable - act - OVERHEAD_GB
    need = weights_gb + act + OVERHEAD_GB
    if hw.vram_gb and tf and room > 0.5:
        frac = 1.0 if weights_gb <= room else max(0.0, room / weights_gb)
        eff = tf * GEN_EFF * (1.0 if is_nvidia(hw) else OTHER_GPU_EFF) / slow
        step = flop_step / eff + weights_gb * (1 - frac) / PCIE_GBS
        swap = (te_gb + min(weights_gb, room)) / PCIE_GBS if te_gb + weights_gb > room else 0.0
        if experts > 1:  # changement d'expert à mi-parcours
            swap += min(weights_gb, room) / PCIE_GBS
        total = steps * step + fixed + swap
        possible = weights_gb * (1 - frac) + (experts - 1) * weights_gb <= hw.ram_gb * 0.75
        cpu = False
    else:  # pas de GPU utilisable : calcul sur le processeur
        cpu_tf = (hw.cpu_gflops or 300.0) / 1000
        frac = 0.0
        step = flop_step / (cpu_tf * CPU_EFF) * slow
        total = steps * step + fixed * 8
        possible = weights_gb * experts + te_gb <= hw.ram_gb * 0.75
        cpu = True
        measured = bool(hw.cpu_gflops)
    return ImgEstimate(weights_gb, need, frac, step, total if possible else None, steps, possible,
                       need <= hw.free_now_gb, prof["video"], cpu, measured)


def entries(cat: dict, hw: ap.Hardware, installed_keys: set[str] | None = None) -> list[dict]:
    installed_keys = installed_keys or set()
    out = []
    for fam in cat.get("families", []):
        if not fam.get("params_b"):
            continue
        prof = profile_for(fam)
        p = prof["params"]
        k = _key(fam["id"])
        use = usages(fam, prof)
        desc = describe(fam, prof)
        for fmt, gb, slow in formats(prof, p):
            dit_gb = 2.0 * p if prof.get("checkpoint") else gb
            out.append({
                "family": fam, "key": k, "name": fam["name"], "format": fmt, "size_gb": gb * prof.get("experts", 1),
                "params_b": p, "profile": prof, "usages": use, "description": desc,
                "estimate": estimate(hw, prof, p, dit_gb, slow),
                "downloads": fam.get("downloads") or 0, "installed": k in installed_keys,
            })
    return out


QUALITY = {"BF16": 5, "FP16": 5, "FP8": 4, "GGUF Q8_0": 4, "GGUF Q5_K_M": 3, "GGUF Q4_K_M": 2}


def best_per_family(items: list[dict]) -> list[dict]:
    """Pour chaque modèle : le format le plus fidèle qui tient en VRAM (à qualité égale, le plus rapide),
    sinon le plus rapide."""
    by: dict[str, list[dict]] = {}
    for i in items:
        by.setdefault(i["key"], []).append(i)

    def rank(i):
        return QUALITY.get(i["format"], 0), -(i["estimate"].total_s or 1e12)
    out = []
    for group in by.values():
        ok = [i for i in group if i["estimate"].possible]
        if not ok:
            out.append(min(group, key=lambda i: i["size_gb"]))
            continue
        full = [i for i in ok if i["estimate"].gpu_frac >= 0.999 or i["estimate"].cpu]
        if full:
            out.append(max(full, key=rank))
        else:
            out.append(min(ok, key=lambda i: i["estimate"].total_s or 1e12))
    return out


# --------------------------------------------------------------------------- modèles installés (ComfyUI)

MODEL_DIRS = ("checkpoints", "diffusion_models", "unet")


def comfy_base_paths() -> list[str]:
    paths = []
    cfg = os.path.join(os.environ.get("APPDATA", ""), "ComfyUI", "config.json")
    try:
        with open(cfg, encoding="utf-8") as f:
            base = json.load(f).get("basePath")
        if base:
            paths.append(base)
    except (OSError, ValueError):
        pass
    paths.append(os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI"))
    return [p for p in paths if os.path.isdir(os.path.join(p, "models"))]


def local_model_files() -> list[str]:
    files = []
    for base in comfy_base_paths():
        for d in MODEL_DIRS:
            folder = os.path.join(base, "models", d)
            try:
                for root, _dirs, names in os.walk(folder):
                    files += [n for n in names if n.lower().endswith((".safetensors", ".gguf", ".ckpt", ".sft"))]
            except OSError:
                pass
    return files


def installed_keys(cat: dict, files: list[str]) -> set[str]:
    """Associe chaque fichier au modèle dont le nom (ou un alias) y figure le plus longuement."""
    keys = {}
    for fam in cat.get("families", []):
        k = _key(fam["id"])
        for alias in [k] + ALIASES.get(k, []):
            if len(alias) >= 4:
                keys.setdefault(alias, k)
    found = set()
    for f in files:
        fk = norm(os.path.splitext(os.path.basename(f))[0])
        match = max((a for a in keys if a in fk), key=len, default=None)
        if match:
            found.add(keys[match])
    return found


# --------------------------------------------------------------------------- résumé (onglet IA, rapport)

SUMMARY_KEYS = ["stablediffusionv15", "stablediffusionxlbase10", "zimageturbo", "flux1dev", "qwenimage",
                "flux2dev", "wan22ti2v5b", "wan22t2va14b"]


def capacity(hw: ap.Hardware, cat: dict | None = None) -> list[tuple[str, str, int]]:
    """Quelques modèles de référence : (nom, « ≈ 12 s / image · 100 % GPU · FP8 », niveau 0-4 ou -1)."""
    cat = cat or load()
    items = best_per_family([i for i in entries(cat, hw) if i["key"] in SUMMARY_KEYS])
    items.sort(key=lambda i: SUMMARY_KEYS.index(i["key"]))
    out = []
    for i in items:
        e = i["estimate"]
        unit = "clip de 5 s" if e.video else "image"
        if e.possible:
            txt = f"≈ {duration(e.total_s)} par {unit} · {e.placement} · {i['format']}"
        else:
            txt = "trop gros pour ce PC"
        out.append((i["name"], txt, e.level))
    return out
