@echo off
rem Compile le module natif (DLL autonome, sans dependance MinGW a l'execution).
rem Necessite g++ (MSYS2 MinGW-w64) dans le PATH.
setlocal
cd /d "%~dp0"
if not exist tflops_meter\bin mkdir tflops_meter\bin
g++ -O2 -std=c++17 -shared -static -static-libgcc -static-libstdc++ ^
    -o tflops_meter\bin\tflops_native.dll native\tflops_native.cpp
if errorlevel 1 (
    echo Echec de la compilation.
    exit /b 1
)
echo OK : tflops_meter\bin\tflops_native.dll
