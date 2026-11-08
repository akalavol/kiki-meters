# TFLOPS Meter

Mesure la puissance de calcul réelle d'un PC Windows : CPU, GPU, RAM et VRAM, avec un test de stabilité, un historique et l'export des résultats.

## Lancer

Double-cliquez sur **`TflopsMeter.cmd`**. Au premier lancement, il installe PySide6, puis cuBLAS si une carte NVIDIA est présente (environ 600 Mo, une seule fois), et compile le module natif s'il manque.

Mode console : `TflopsMeter.cmd --cli --preset Rapide --only gpu cpu`

Prérequis : Python 3.10 ou plus récent. Le module natif (tests CPU et RAM) n'est pas fourni compilé dans ce dépôt : `TflopsMeter.cmd` le compile au premier lancement si `g++` (MSYS2 MinGW-w64) est dans le PATH, sinon lancez `build_native.cmd` après l'avoir installé.

## Ce qui est mesuré

| Composant | Tests | Unité |
|---|---|---|
| CPU | FP32 / FP64, monocœur et multicœur (noyaux FMA AVX2 en C++) | GFLOPS |
| GPU NVIDIA | GEMM cuBLAS FP32, TF32, FP16, BF16, FP64, plus la crête FMA FP32 | TFLOPS |
| GPU AMD / Intel | Crête FMA FP32 / FP16 / FP64 (OpenCL) | TFLOPS |
| RAM | Lecture, écriture, copie (multithread) | Go/s |
| RAM | Latence en accès aléatoires (chaînage de pointeurs) | ns |
| VRAM | Bande passante de copie GPU → GPU | Go/s |
| Stockage | Pour chaque disque : lecture et écriture séquentielles (1 Mo, file de 8), aléatoire 4K en Q1 et lecture 4K en Q32, en E/S directes sans cache Windows. Le fichier de test est supprimé à la fin. | Mo/s, IOPS |
| Stabilité | Charge continue de 1 à 180 min, avec la performance, la température, la fréquence, la puissance et les causes de bridage | courbes et bilan |

Durées : **Rapide** (~0,7 s par test), **Standard** (~2 s), **Précis** (~5 s). Le meilleur passage est retenu.

L'application signale si le GPU est déjà occupé (jeu, IA, rendu) au lancement d'une analyse. Dans ce cas, les résultats seraient sous-estimés.

## IA locale

TFLOPS Meter a été pensé pour aider une IA (Claude, ChatGPT…) à vous conseiller sur l'IA locale. L'onglet **IA locale** fonctionne sur **tous les PC**, y compris sans outil d'IA installé et sans carte graphique dédiée :

- **Capacité du PC** : quelles tailles de modèle (3B à 70B, MoE) tiennent en VRAM, dans quelles quantifications (Q4 / Q5 / Q8), avec quel contexte, à quelle vitesse estimée.
- **Modèles installés** : détection d'Ollama, de LM Studio et de ComfyUI. Pour chaque modèle Ollama, l'onglet donne la mémoire réelle (poids + cache KV calculé depuis l'architecture), le placement GPU / RAM et le contexte maximal en 100 % GPU.
- **Mesure réelle** : génération et lecture du prompt (tok/s), plus la part réellement chargée en VRAM, via l'API d'Ollama.
- **⚡ Mesure rapide** : RAM, VRAM et FP16 en environ 15 s, pour baser les estimations sur ce PC.
- **📋 Copier le rapport pour Claude / ChatGPT** : un profil complet en Markdown (matériel, mesures, modèles, constats et demande toute prête), à coller dans une conversation.
- **🖼 Image pour Discord** : la configuration (CPU, GPU, RAM avec type et fréquence, carte mère, disques, OS) et les performances mesurées (calcul, mémoire, stockage), en JPG, ou **Copier l'image** puis Ctrl+V dans Discord.

L'onglet **Modèles compatibles** liste tous les modèles de la bibliothèque Ollama (≈ 270 modèles, plus de 500 variantes), évalués pour ce PC. Pour chacun, il affiche :
- les usages (discussion, code, vision, raisonnement, agents, embeddings) ;
- la taille et le contexte maximal ;
- le placement estimé (VRAM ou RAM) et la vitesse estimée, avec un verdict.

Des filtres permettent de chercher par nom, de choisir un usage, de n'afficher que les modèles qui tournent bien, et de ne garder qu'une taille par modèle. Depuis cet onglet, vous pouvez **installer** le modèle avec Ollama (avec une barre de progression), copier la commande, ou chercher sa version GGUF pour LM Studio. Une liste est intégrée au logiciel, et **⟳ Mettre à jour la liste** télécharge celle du jour depuis ollama.com (≈ 15 s).

Le sous-onglet **Images et vidéo** fait de même pour la génération d'images et de vidéos (≈ 70 modèles les plus téléchargés sur Hugging Face : Flux, Qwen-Image, Z-Image, SDXL, Wan, LTX…). Pour chaque modèle et chaque format (BF16, FP8, GGUF Q8 / Q5 / Q4), il affiche :
- la taille et la VRAM nécessaire, avec le placement (100 % GPU, mixte ou CPU) ;
- le temps estimé par image en 1024×1024, ou par clip vidéo de 5 s, avec un verdict.

Les modèles déjà présents dans ComfyUI sont repérés. Des liens ouvrent la page du modèle, les fichiers prêts pour ComfyUI et la version GGUF. **⟳ Mettre à jour la liste** interroge l'API de Hugging Face en quelques secondes. Le temps par image est estimé ainsi : nombre d'étapes × calcul d'une étape ÷ puissance FP16 mesurée, plus le transfert PCIe quand le modèle déborde de la VRAM. C'est un ordre de grandeur, calibré sur des temps SDXL et Flux connus.

Les estimations de vitesse suivent la règle des LLM locaux : tok/s ≈ bande passante mémoire ÷ taille du modèle. Elles sont calibrées sur des mesures Ollama réelles, avec un écart de 2 à 5 % quand le modèle est entièrement en VRAM. Quand un modèle déborde en RAM, l'estimation est approximative : mesurez-le.

## Onglets

- **Analyse** : choix des tests, résultats en direct, cartes de synthèse, graphiques mesuré / théorique, export.
- **IA locale** : voir ci-dessus.
- **Stabilité** : charge prolongée avec courbes en temps réel et verdict (stable, légère baisse, bridage).
- **Historique** : toutes les mesures (SQLite), détail, comparaison de deux mesures, renommage, export JSON / CSV, suppression.
- **Paramètres** : threads, tailles de tampon, précisions GPU, taille des matrices, fréquence de surveillance.

La barre d'état affiche en permanence la température, la fréquence, la puissance et la charge du GPU (NVML), ainsi que la fréquence et la charge du CPU (compteurs Windows).

Données : `%LOCALAPPDATA%\TflopsMeter\` (`historique.db`, `parametres.json`).

## Architecture

| Fichier | Rôle |
|---|---|
| `native/tflops_native.cpp` | DLL C++ : FMA AVX2/FMA (repli SSE2), débit RAM (AVX2 + écritures non temporelles), latence |
| `tflops_meter/cuda.py` | Pilote CUDA (`nvcuda.dll`) et cuBLAS via ctypes : GEMM, VRAM |
| `tflops_meter/opencl.py`, `gpu_opencl.py` | OpenCL via ctypes : crête FMA et VRAM pour les GPU non NVIDIA |
| `tflops_meter/monitor.py` | NVML (température, fréquences, puissance, bridage) et PDH (fréquence CPU) |
| `tflops_meter/runner.py` | Orchestration de l'analyse et du test de stabilité |
| `tflops_meter/history.py` | Historique SQLite, export JSON / CSV |
| `tflops_meter/ui/` | Interface PySide6 / QtCharts |

Le GPU NVIDIA est piloté sans CUDA Toolkit ni nvcc. Le pilote CUDA est fourni avec tout pilote NVIDIA, et cuBLAS vient du paquet pip `nvidia-cublas-cu12`.

## Construire

- Module natif : `build_native.cmd` (g++ MinGW-w64, DLL autonome liée statiquement).
- Application portable : `build_exe.cmd` produit `dist\TflopsMeter\TflopsMeter.exe`, cuBLAS compris (~700 Mo). `build_exe.cmd leger` produit une version sans cuBLAS (~150 Mo).

## Limites connues

- La température CPU n'est pas exposée par Windows sans pilote tiers. Elle s'affiche donc « non disponible ».
- Les TFLOPS théoriques supposent la fréquence boost et des FMA parfaites. Une GEMM réelle atteint 70 à 90 % de cette valeur. Les cartes qui boostent au-delà de la fréquence annoncée peuvent dépasser 100 % sur la crête FMA.
- La latence RAM est mesurée avec des pages de 4 Ko, défauts TLB compris. Elle est donc plus élevée que les valeurs obtenues avec de grandes pages.
- FP16 sur GPU AMD / Intel : dépend de l'extension `cl_khr_fp16` du pilote.
