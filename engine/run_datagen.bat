@echo off
chcp 65001 >nul
echo ==========================================
echo   Togyz Kumalaq - NNUE Data Generation
echo ==========================================
echo.

:: Configuration - adjust these as needed
:: 35000 games × ~150 pos/game = ~5M positions per machine
:: 10 machines × 5M = ~50M total
set GAMES=35000
set DEPTH=10
set THREADS=8
set PREFIX=%COMPUTERNAME%

echo Machine: %PREFIX%
echo Games: %GAMES%
echo Depth: %DEPTH%
echo Threads: %THREADS%
echo.

:: Check that engine and NNUE weights exist
if not exist togyzkumalaq-engine.exe (
    echo ERROR: togyzkumalaq-engine.exe not found!
    echo Place it in the same directory as this script.
    pause
    exit /b 1
)

if not exist nnue_weights.bin (
    echo WARNING: nnue_weights.bin not found!
    echo Will use handcrafted eval (weaker).
    echo.
)

echo Starting data generation...
echo Output: %PREFIX%_training_data.bin
echo.

togyzkumalaq-engine.exe datagen %GAMES% %DEPTH% %THREADS% %PREFIX%

echo.
echo ==========================================
echo Done! Copy %PREFIX%_training_data.bin back.
echo ==========================================
pause
