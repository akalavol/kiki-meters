@echo off
rem Lance TFLOPS Meter. Au premier lancement : installe les dependances et compile le module natif.
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python 3.10 ou plus recent est necessaire : https://www.python.org/downloads/
    pause
    exit /b 1
)

python -c "import PySide6" >nul 2>nul
if errorlevel 1 (
    echo Installation de l'interface PySide6...
    python -m pip install --user "PySide6>=6.6" || goto :erreur
)

rem cuBLAS (GEMM sur Tensor Cores) seulement si une carte NVIDIA est presente.
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    python -c "import nvidia.cublas" >nul 2>nul
    if errorlevel 1 (
        echo Carte NVIDIA detectee : installation de cuBLAS ^(environ 600 Mo, une seule fois^)...
        python -m pip install --user nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
    )
)

if not exist tflops_meter\bin\tflops_native.dll (
    where g++ >nul 2>nul
    if errorlevel 1 (
        echo Module natif absent et g++ introuvable : les tests CPU et RAM seront indisponibles.
    ) else (
        call build_native.cmd
    )
)

if "%~1"=="--cli" (
    python -m tflops_meter %*
    pause
    exit /b
)
start "" pythonw -m tflops_meter
exit /b 0

:erreur
echo Echec de l'installation des dependances.
pause
exit /b 1
