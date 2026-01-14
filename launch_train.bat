@echo off
echo ========================================
echo   Aventa HFT Pro 2026 - ML Training
echo ========================================
echo.

cd /d "%~dp0"

python -c "from ml_predictor import MLPredictor; import MetaTrader5 as mt5; mt5.initialize(); predictor = MLPredictor('EURUSD', {}); predictor.train(30); predictor.save_models('./models'); mt5.shutdown(); print('\nTraining completed!')"

echo.
pause
