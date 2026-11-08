# -*- mode: python ; coding: utf-8 -*-
# Construction : build_exe.cmd  (ou : pyinstaller --noconfirm TflopsMeter.spec)
# Variable d'environnement TFLOPS_NO_CUBLAS=1 pour un paquet léger sans cuBLAS (~600 Mo de moins).
import glob
import os

datas = [("tflops_meter/bin/tflops_native.dll", "tflops_meter/bin"),
         ("tflops_meter/data/catalogue.json", "tflops_meter/data"),
         ("tflops_meter/data/catalogue_images.json", "tflops_meter/data")]

if os.environ.get("TFLOPS_NO_CUBLAS") != "1":
    try:
        import nvidia.cublas as _cb
        for p in _cb.__path__:
            for dll in glob.glob(os.path.join(p, "bin", "cublas*64_*.dll")):
                datas.append((dll, "tflops_meter/bin/cuda"))
    except ImportError:
        print("cuBLAS non installé : l'exécutable mesurera la crête FMA OpenCL à la place des GEMM.")
    try:
        import nvidia.cuda_runtime as _rt
        for p in _rt.__path__:
            for dll in glob.glob(os.path.join(p, "bin", "cudart64_*.dll")):
                datas.append((dll, "tflops_meter/bin/cuda"))
    except ImportError:
        pass

a = Analysis(
    ["run.py"],
    pathex=[],
    datas=datas,
    hiddenimports=["PySide6.QtCharts"],
    # « nvidia » exclu : ses DLL sont déjà copiées ci-dessus dans tflops_meter/bin/cuda (sinon doublon de 735 Mo)
    excludes=["nvidia", "tkinter", "numpy", "torch", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
              "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtMultimedia"],
    noarchive=False,
)
# L'analyse des dépendances de cublas64_12.dll ramène cublasLt64_12.dll (638 Mo) une seconde fois sous
# nvidia/ : on ne garde que la copie de tflops_meter/bin/cuda (chargée depuis ce dossier à l'exécution).
a.binaries = [b for b in a.binaries if not b[0].replace("\\", "/").lower().startswith("nvidia/")]
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TflopsMeter",
    console=False,
    upx=False,
    icon=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="TflopsMeter")
