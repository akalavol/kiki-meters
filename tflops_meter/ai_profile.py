"""Capacité « IA locale » d'un PC : quels modèles tiennent en VRAM, à quelle vitesse, avec quels réglages.

Modèle d'estimation (celui qu'utilisent llama.cpp / Ollama en pratique) :
- mémoire = poids du modèle + cache KV (dépend du contexte) + tampons de calcul ;
- la génération est limitée par la bande passante : chaque token relit les poids actifs,
  donc tok/s ≈ bande passante utile ÷ taille des poids actifs (VRAM et RAM si débordement) ;
- le traitement du prompt (prefill) est limité par la puissance FP16 des Tensor Cores.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from . import __version__

# Coefficients calibrés sur des mesures Ollama réelles (RTX 5060 Ti, i5-14400F, nov. 2026) :
# mistral 7B Q4 82 tok/s, qwen3 14B Q4 42 tok/s, MoE 27B-A3B à 86 % en VRAM 14,5 tok/s.
RESERVE_GB = 0.8      # Windows, bureau, pilote
OVERHEAD_GB = 0.6     # contexte CUDA + tampons de calcul de llama.cpp
GPU_EFF = 0.98        # relatif à la bande passante de copie mesurée (les noyaux lisent mieux qu'une copie)
CPU_EFF = 0.60        # couches denses déchargées en RAM : limitées par la bande passante
CPU_EFF_MOE = 0.15    # experts MoE sur CPU : routage et petits produits, bien moins efficaces
PREFILL_EFF = 0.80    # relatif à la GEMM FP16 cuBLAS mesurée

QUANT_BITS = {"Q4_K_M": 4.85, "Q5_K_M": 5.69, "Q8_0": 8.5}

# Classes de modèles génériques : (nom, paramètres totaux en milliards, actifs, octets de cache KV par token)
MODEL_CLASSES = [
    ("3-4B", 3.8, 3.8, 147_456),
    ("7-8B", 8.0, 8.0, 131_072),
    ("12-14B", 14.0, 14.0, 163_840),
    ("22-24B", 24.0, 24.0, 163_840),
    ("27-32B", 32.0, 32.0, 262_144),
    ("70B", 70.0, 70.0, 327_680),
    ("MoE 30B (3B actifs)", 30.5, 3.3, 98_304),
    ("MoE 120B (5B actifs)", 117.0, 5.1, 73_728),
]

def fr(v, dec=1):
    if v is None:
        return "—"
    return f"{v:,.{dec}f}".replace(",", " ").replace(".", ",")


@dataclass
class Hardware:
    cpu: str = ""
    cores: int | None = None
    threads: int | None = None
    ram_gb: float = 16.0
    ram_bw: float = 40.0
    ram_bw_src: str = "valeur par défaut"
    gpu: str = ""
    vram_gb: float = 0.0
    vram_used_now_gb: float = 0.0
    vram_bw: float | None = None
    vram_bw_src: str = ""
    fp16_tflops: float | None = None
    fp32_tflops: float | None = None
    bf16_tflops: float | None = None
    cpu_gflops: float | None = None
    cc: str | None = None
    driver: str | None = None
    extras: dict = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        """Vrai si les débits utilisés par les estimations viennent de vraies mesures."""
        return self.ram_bw_src == "mesurée" and (not self.gpu or self.vram_bw_src == "mesurée")

    @property
    def usable_gb(self):
        return max(0.0, self.vram_gb - RESERVE_GB)

    @property
    def free_now_gb(self):
        return max(0.0, self.vram_gb - self.vram_used_now_gb - RESERVE_GB / 2)


def build_hardware(sysinfo: dict, measures: list[dict], gpu_sample: dict | None, driver: str | None = None,
                   theoretical_vram_bw: float | None = None) -> Hardware:
    """measures : dicts au format Measure (dernière analyse)."""
    def find(cat, test_part):
        for m in measures:
            if m.get("category") == cat and test_part in m.get("test", "") and m.get("value") and not m.get("error"):
                return m
        return None
    hw = Hardware(cpu=sysinfo.get("cpu", ""), cores=sysinfo.get("cores"), threads=sysinfo.get("threads"),
                  ram_gb=sysinfo.get("ram_gb") or 16.0, driver=driver)
    r = find("RAM", "Lecture")
    if r:
        hw.ram_bw, hw.ram_bw_src = r["value"], "mesurée"
    c = find("CPU", "FP32 multicœur")
    if c:
        hw.cpu_gflops = c["value"]
    gpus = sysinfo.get("gpus") or []
    dedicated = [g for g in gpus if not g.get("integrated")]
    igpu = [g for g in gpus if g.get("integrated")]
    if igpu:
        hw.extras["igpu"] = igpu[0]["name"]
    if dedicated:
        g = max(dedicated, key=lambda x: x.get("memory_gb") or 0)
        hw.gpu, hw.vram_gb, hw.cc = g["name"], g.get("memory_gb") or 0.0, g.get("compute_capability")
        v = find("VRAM", "Bande passante")
        if v:
            hw.vram_bw, hw.vram_bw_src = v["value"], "mesurée"
        elif theoretical_vram_bw:
            hw.vram_bw, hw.vram_bw_src = theoretical_vram_bw * 0.85, "théorique × 0,85 (lancez la mesure)"
        else:  # ordre de grandeur d'une carte milieu de gamme, à remplacer par une mesure
            hw.vram_bw, hw.vram_bw_src = 250.0, "valeur par défaut, non mesurée (lancez la mesure)"
        for prec, attr in (("FP16", "fp16_tflops"), ("FP32", "fp32_tflops"), ("BF16", "bf16_tflops")):
            m = find("GPU", f"GEMM {prec}")
            if m:
                setattr(hw, attr, m["value"])
        for prec in ("FP16", "FP32"):  # GPU AMD / Intel : crête FMA mesurée en OpenCL
            m = find("GPU", f"Crête FMA {prec}")
            if m:
                hw.extras[f"fma_{prec.lower()}"] = m["value"]
    if gpu_sample and gpu_sample.get("mem_used_gb") is not None:
        hw.vram_used_now_gb = gpu_sample["mem_used_gb"]
    return hw


def parse_params(s: str) -> float | None:
    m = re.match(r"([\d.]+)\s*([BMK])", s or "", re.I)
    if not m:
        return None
    v = float(m.group(1))
    return {"B": v, "M": v / 1000, "K": v / 1e6}[m.group(2).upper()]


@dataclass
class Estimate:
    weights_gb: float
    kv_gb: float
    total_gb: float
    gpu_frac: float
    gen_tps: float | None
    prefill_tps: float | None
    max_ctx_gpu: int | None
    possible: bool
    fits_now: bool

    @property
    def placement(self):
        if not self.possible:
            return "trop gros"
        if self.gpu_frac >= 0.999:
            return "100 % GPU"
        if self.gpu_frac <= 0.001:
            return "CPU"
        return f"mixte ({self.gpu_frac * 100:.0f} % GPU)"

    @property
    def verdict(self):
        t = self.gen_tps
        if not self.possible or t is None:
            return "impossible"
        return ("très fluide" if t >= 30 else "fluide" if t >= 15 else "utilisable" if t >= 7
                else "lent" if t >= 3 else "très lent")


def estimate(hw: Hardware, weights_gb: float, active_gb: float, kv_per_token: int | None, ctx: int,
             active_params_b: float | None = None, model_max_ctx: int | None = None,
             moe: bool = False) -> Estimate:
    kv_per_token = kv_per_token or int(weights_gb * 1e9 / 60_000)  # repli grossier si architecture inconnue
    kv_gb = kv_per_token * ctx / 1e9
    total = weights_gb + kv_gb + OVERHEAD_GB
    usable = hw.usable_gb
    possible = total <= usable + hw.ram_gb * 0.75
    if not hw.vram_gb:
        frac = 0.0
    elif total <= usable:
        frac = 1.0
    else:
        frac = max(0.0, min(1.0, (usable - OVERHEAD_GB) / (weights_gb + kv_gb)))
    gen = None
    if possible:
        t = 0.0
        if frac > 0 and hw.vram_bw:
            t += active_gb * frac / (hw.vram_bw * GPU_EFF)
        t += active_gb * (1 - frac) / (hw.ram_bw * (CPU_EFF_MOE if moe else CPU_EFF))
        gen = 1 / t if t > 0 else None
    pp = None
    if frac >= 0.999 and hw.fp16_tflops and active_params_b:
        pp = hw.fp16_tflops * 1e12 * PREFILL_EFF / (2 * active_params_b * 1e9)
    max_ctx = None
    room = usable - weights_gb - OVERHEAD_GB
    if room > 0 and kv_per_token:
        max_ctx = int(room * 1e9 / kv_per_token)
        if model_max_ctx:
            max_ctx = min(max_ctx, model_max_ctx)
    fits_now = total <= hw.free_now_gb
    return Estimate(weights_gb, kv_gb, total, frac, gen, pp, max_ctx, possible, fits_now)


def class_matrix(hw: Hardware, ctx: int) -> list[tuple[str, dict[str, Estimate]]]:
    rows = []
    for name, params, active, kv in MODEL_CLASSES:
        row = {}
        for q, bits in QUANT_BITS.items():
            w = params * bits / 8
            a = active * bits / 8
            row[q] = estimate(hw, w, a, kv, ctx, active_params_b=active, moe=active < params)
        rows.append((name, row))
    return rows


def installed_estimates(hw: Hardware, models: list[dict], ctx: int) -> list[tuple[dict, Estimate | None]]:
    out = []
    for m in models:
        if m.get("embedding_only"):
            out.append((m, None))
            continue
        w = (m.get("size_bytes") or 0) / 1e9
        if m.get("experts") and m.get("experts_used"):
            ratio = m["experts_used"] / m["experts"]
            a = w * (0.12 + 0.88 * ratio)
        else:
            a = w
        p = parse_params(m.get("params", ""))
        active_p = p * (a / w) if p and w else p
        use_ctx = min(ctx, m["max_ctx"]) if m.get("max_ctx") else ctx
        out.append((m, estimate(hw, w, a, m.get("kv_bytes_per_token"), use_ctx, active_p, m.get("max_ctx"),
                                moe=bool(m.get("experts")))))
    return out


def image_capacity(hw: Hardware) -> list[tuple[str, str, int]]:
    """Modèles d'image et de vidéo de référence : (nom, estimation, niveau 0-4 ou -1 si impossible)."""
    from . import image_models  # import tardif : image_models dépend de ce module
    return image_models.capacity(hw)


def recommendations(hw: Hardware, ctx: int, tools: dict, installed: list[tuple[dict, Estimate | None]]) -> list[str]:
    rec = []
    if not hw.measured:
        rec.append("Estimations provisoires : cliquez sur « Mesure rapide pour l'IA » (≈ 15 s) pour les "
                   "calculer à partir des vrais débits de ce PC.")
    if not hw.vram_gb:
        why = (f"le GPU intégré ({hw.extras['igpu']}) partage la RAM" if hw.extras.get("igpu")
               else "aucune carte graphique dédiée détectée")
        rec.append(f"Pas de VRAM dédiée ({why}) : les modèles tournent sur le processeur et la vitesse dépend "
                   f"de la RAM ({fr(hw.ram_bw)} Go/s, {hw.ram_bw_src}).")
        matrix = dict(class_matrix(hw, ctx))
        ok = [n for n, *_ in MODEL_CLASSES if (matrix[n]["Q4_K_M"].gen_tps or 0) >= 7]
        if ok:
            n = ok[-1]
            rec.append(f"Taille conseillée : jusqu'à la classe {n} en Q4_K_M "
                       f"(≈ {fr(matrix[n]['Q4_K_M'].gen_tps, 0)} tok/s, utilisable pour discuter).")
        else:
            rec.append("Même les petits modèles seront lents : préférez la classe 3-4B en Q4_K_M, "
                       "ou les assistants en ligne pour les tâches lourdes.")
        moe = matrix["MoE 30B (3B actifs)"]["Q4_K_M"]
        if moe.possible:
            rec.append(f"Bon compromis sans GPU : un MoE 30B à 3B actifs (≈ {fr(moe.gen_tps, 0)} tok/s) donne des "
                       f"réponses de bien meilleure qualité qu'un petit modèle, s'il tient en RAM "
                       f"({fr(moe.total_gb)} Go sur {fr(hw.ram_gb, 0)} Go).")
        rec.append("Génération d'images : très lente sans carte graphique dédiée (plusieurs minutes par image).")
        rec.extend(getting_started(hw, tools))
        return rec
    rec.append(f"{fr(hw.vram_gb)} Go de VRAM, dont ≈ {fr(hw.usable_gb)} Go utilisables pour un modèle "
               f"(bande passante {fr(hw.vram_bw, 0)} Go/s, {hw.vram_bw_src}).")
    if hw.vram_used_now_gb > 1.5:
        who = " (ComfyUI est ouvert)" if tools.get("comfyui", {}).get("running") else ""
        rec.append(f"{fr(hw.vram_used_now_gb)} Go de VRAM sont déjà occupés en ce moment{who}. Libérez-les "
                   "avant de lancer un LLM, sinon une partie du modèle bascule en RAM et la vitesse s'effondre.")
    matrix = dict(class_matrix(hw, ctx))
    dense = [n for n, *_ in MODEL_CLASSES if not n.startswith("MoE")]
    best_q4 = [n for n in dense if matrix[n]["Q4_K_M"].gpu_frac >= 0.999]
    if best_q4:
        n = best_q4[-1]
        e = matrix[n]["Q4_K_M"]
        rec.append(f"Plus gros modèle classique entièrement en VRAM avec {ctx // 1024}k de contexte : "
                   f"classe {n} en Q4_K_M ({fr(e.total_gb)} Go, ≈ {fr(e.gen_tps, 0)} tok/s).")
    best_q8 = [n for n in dense if matrix[n]["Q8_0"].gpu_frac >= 0.999]
    if best_q8:
        n = best_q8[-1]
        rec.append(f"En qualité quasi maximale (Q8_0), la classe {n} tient en VRAM "
                   f"(≈ {fr(matrix[n]['Q8_0'].gen_tps, 0)} tok/s).")
    moe = matrix.get("MoE 30B (3B actifs)", {}).get("Q4_K_M")
    if moe and moe.possible:
        rec.append(f"Les modèles MoE restent utilisables même s'ils débordent en RAM : un MoE 30B à 3B actifs "
                   f"donne ≈ {fr(moe.gen_tps, 0)} tok/s ({moe.placement}), contre quelques tok/s pour un "
                   "modèle dense de même taille. Estimation grossière : mesurez-le.")
    big = matrix["70B"]["Q4_K_M"]
    if big.possible and big.gpu_frac < 1:
        rec.append(f"Un 70B en Q4 est faisable grâce aux {fr(hw.ram_gb, 0)} Go de RAM, mais lent "
                   f"(≈ {fr(big.gen_tps)} tok/s, {big.placement}).")
    over = [(m, e) for m, e in installed if e and e.gpu_frac < 0.999 and e.possible]
    for m, e in over:
        rec.append(f"« {m['name']} » ({fr(e.weights_gb)} Go) déborde de la VRAM à {ctx // 1024}k de contexte : "
                   f"{e.placement}, ≈ {fr(e.gen_tps, 0)} tok/s. Solutions : contexte plus court, quantification "
                   "plus légère (IQ4_XS, Q3_K_M) ou cache KV en q8_0.")
    kv_example = next((e for m, e in installed if e and e.kv_gb > 0.5), None)
    if kv_example:
        rec.append(f"Le cache KV coûte cher : jusqu'à {fr(kv_example.kv_gb)} Go pour {ctx // 1024}k tokens sur vos "
                   "modèles. Avec Ollama, OLLAMA_FLASH_ATTENTION=1 et OLLAMA_KV_CACHE_TYPE=q8_0 divisent ce coût "
                   "par deux pour une perte de qualité négligeable.")
    rec.append("La génération dépend surtout de la bande passante mémoire : vitesse ≈ bande passante ÷ taille "
               "du modèle. Un modèle deux fois plus petit (ou mieux quantifié) va deux fois plus vite.")
    if tools.get("ollama", {}).get("running"):
        rec.append("Dans Ollama, réglez num_ctx explicitement (paramètre du Modelfile ou de l'API) : un contexte "
                   "trop grand fait déborder le modèle en RAM sans prévenir. Vérifiez avec « ollama ps » "
                   "(colonne PROCESSOR = 100% GPU).")
    rec.extend(getting_started(hw, tools))
    return rec


def has_ai_tools(tools: dict) -> bool:
    return any(tools.get(k, {}).get("running") or tools.get(k, {}).get("installed")
               for k in ("ollama", "lmstudio", "comfyui"))


def getting_started(hw: Hardware, tools: dict) -> list[str]:
    """Conseils de démarrage pour un PC sans outil d'IA locale (ou avec Ollama arrêté)."""
    o = tools.get("ollama", {})
    if o.get("installed") and not o.get("running"):
        return ["Ollama est installé mais arrêté : lancez-le (menu Démarrer), puis cliquez sur « Actualiser » "
                "pour voir vos modèles et mesurer leur vitesse réelle."]
    if has_ai_tools(tools):
        return []
    matrix = dict(class_matrix(hw, 8192))
    if hw.vram_gb:
        fits = [n for n, *_ in MODEL_CLASSES if not n.startswith("MoE") and matrix[n]["Q4_K_M"].gpu_frac >= 0.999]
        first = fits[-1] if fits else "3-4B"
    else:
        ok = [n for n, *_ in MODEL_CLASSES if (matrix[n]["Q4_K_M"].gen_tps or 0) >= 7]
        first = ok[-1] if ok else "3-4B"
    tips = [
        "Aucun outil d'IA locale détecté. Pour commencer : Ollama (ollama.com, en ligne de commande, idéal pour "
        "les assistants et le code) ou LM Studio (lmstudio.ai, interface graphique pour essayer des modèles).",
        f"Premier modèle conseillé pour ce PC : un modèle de la classe {first} en Q4_K_M. Une fois installé, "
        "revenez ici et cliquez sur « Actualiser » pour mesurer sa vitesse réelle.",
    ]
    if hw.vram_gb >= 6:
        tips.append("Pour la génération d'images : ComfyUI (comfy.org), avec SDXL en premier modèle.")
    return tips


# --------------------------------------------------------------------------- rapport pour IA

def build_report(hw: Hardware, tools: dict, ctx: int, installed, measured: dict, measures: list[dict]) -> str:
    L = []
    L.append(f"# Profil matériel pour l'IA locale — TFLOPS Meter {__version__}")
    L.append(f"_Généré le {datetime.now():%d/%m/%Y %H:%M}. Toutes les valeurs « mesurées » viennent de tests réels "
             "sur cette machine ; les « estimations » sont calculées à partir de ces mesures._\n")
    L.append("## Matériel")
    L.append(f"- **CPU** : {hw.cpu} — {hw.cores or '?'} cœurs / {hw.threads or '?'} threads")
    L.append(f"- **RAM** : {fr(hw.ram_gb, 0)} Go, débit lecture {fr(hw.ram_bw)} Go/s ({hw.ram_bw_src})")
    if hw.gpu:
        L.append(f"- **GPU** : {hw.gpu} — {fr(hw.vram_gb)} Go de VRAM"
                 + (f", compute capability {hw.cc}" if hw.cc else "") + (f", pilote {hw.driver}" if hw.driver else ""))
        L.append(f"- **VRAM** : bande passante {fr(hw.vram_bw, 0)} Go/s ({hw.vram_bw_src}) ; "
                 f"{fr(hw.vram_used_now_gb)} Go déjà occupés au moment du rapport")
    else:
        L.append("- **GPU** : aucune carte graphique dédiée"
                 + (f" (GPU intégré {hw.extras['igpu']}, mémoire partagée avec la RAM)" if hw.extras.get("igpu") else ""))
    if not hw.measured:
        L.append("- _Débits non mesurés : estimations approximatives._")
    perf = [m for m in measures if m.get("value") and not m.get("error")]
    if perf:
        L.append("\n## Performances mesurées")
        L.append("| Composant | Test | Résultat |")
        L.append("|---|---|---|")
        for m in perf:
            L.append(f"| {m['category']} | {m['test']} | {fr(m['value'], 2)} {m['unit']} |")
    L.append("\n## Logiciels d'IA locale détectés")
    o, lm, c = tools.get("ollama", {}), tools.get("lmstudio", {}), tools.get("comfyui", {})
    L.append(f"- Ollama : {'version ' + o['version'] + ', en cours' if o.get('running') else ('installé, arrêté' if o.get('installed') else 'non détecté')}")
    L.append(f"- LM Studio : {'serveur actif' if lm.get('running') else ('installé' if lm.get('installed') else 'non détecté')}")
    L.append(f"- ComfyUI : {'en cours (v' + str(c.get('version')) + ')' if c.get('running') else 'non détecté'}")
    if installed:
        L.append(f"\n## Modèles installés (Ollama) — estimation à {ctx // 1024}k de contexte")
        L.append("| Modèle | Paramètres | Quantification | Taille | Contexte max du modèle | Placement estimé | "
                 "Contexte max 100 % GPU | Génération estimée | Génération mesurée | Prompt mesuré | Part en VRAM mesurée |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for m, e in installed:
            b = measured.get(m["name"], {})
            if e is None:
                L.append(f"| {m['name']} | {m['params']} | {m['quant']} | {fr(m['size_bytes'] / 1e9)} Go | "
                         f"{m.get('max_ctx') or '?'} | embeddings | — | — | — | — | — |")
                continue
            moe = f" (MoE {m['experts_used']}/{m['experts']} experts)" if m.get("experts") else ""
            L.append(f"| {m['name']}{moe} | {m['params']} | {m['quant']} | {fr(e.weights_gb)} Go | "
                     f"{m.get('max_ctx') or '?'} | {e.placement} | {fr(e.max_ctx_gpu, 0) if e.max_ctx_gpu else '—'} | "
                     f"{fr(e.gen_tps, 0)} tok/s | {fr(b.get('gen_tps'), 1) + ' tok/s' if b.get('gen_tps') else '—'} | "
                     f"{fr(b.get('prompt_tps'), 0) + ' tok/s' if b.get('prompt_tps') else '—'} | "
                     f"{fr(b.get('gpu_pct'), 0) + ' %' if b.get('gpu_pct') is not None else '—'} |")
    L.append(f"\n## Capacité estimée par taille de modèle ({ctx // 1024}k de contexte)")
    L.append("| Classe | Q4_K_M | Q5_K_M | Q8_0 |")
    L.append("|---|---|---|---|")
    for name, row in class_matrix(hw, ctx):
        cells = []
        for q in QUANT_BITS:
            e = row[q]
            cells.append(f"{fr(e.total_gb)} Go · {e.placement} · ≈ {fr(e.gen_tps, 0)} tok/s" if e.possible
                         else f"{fr(e.total_gb)} Go · trop gros")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("\n## Génération d'images et de vidéos (estimation, ComfyUI, 1024×1024)")
    for name, v, _level in image_capacity(hw):
        L.append(f"- {name} : {v}")
    L.append("\n## Constats automatiques")
    for r in recommendations(hw, ctx, tools, installed):
        L.append(f"- {r}")
    L.append("\n## Ma demande")
    L.append("En te basant sur ce profil, aide-moi à tirer le meilleur de l'IA locale sur ce PC :")
    L.append("1. quels modèles et quelles quantifications choisir pour mes usages (à préciser : discussion, code, "
             "vision, RAG, génération d'images…) ;")
    L.append("2. les réglages précis (num_ctx, couches GPU, flash attention, cache KV, threads) pour Ollama / "
             "llama.cpp / LM Studio ;")
    L.append("3. ce qui limite mes performances et ce qu'il faudrait éviter ou améliorer.")
    L.append("\n_Mes usages : …_")
    return "\n".join(L)
