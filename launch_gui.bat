@echo off
echo ========================================
echo   Aventa HFT Pro 2026 - GUI Launcher
echo ========================================
echo.

cd /d "%~dp0"

python gui_launcher.py

if errorlevel 1 (
    echo.
    echo Error occurred!
    pause
)
