@echo off
rem Construit l'application portable dans dist\TflopsMeter\ (dossier a copier tel quel).
rem   build_exe.cmd           : avec cuBLAS (GEMM Tensor Cores sur NVIDIA), ~700 Mo
rem   build_exe.cmd leger     : sans cuBLAS, ~150 Mo (crete FMA OpenCL a la place)
setlocal
cd /d "%~dp0"
if /i "%~1"=="leger" set TFLOPS_NO_CUBLAS=1

if not exist tflops_meter\bin\tflops_native.dll call build_native.cmd || exit /b 1
python -c "import PyInstaller" 2>nul || python -m pip install --user pyinstaller || exit /b 1
python -m PyInstaller --noconfirm --clean TflopsMeter.spec || exit /b 1
echo.
echo Application prete : dist\TflopsMeter\TflopsMeter.exe
