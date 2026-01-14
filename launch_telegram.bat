@echo off
echo ========================================
echo  Aventa HFT Pro 2026 - Telegram Bot
echo ========================================
echo.

cd /d "%~dp0"

python telegram_bot.py

if errorlevel 1 (
    echo.
    echo Error occurred!
    pause
)
