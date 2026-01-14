# 🎯 Quick Start Guide

## Instalasi Cepat (5 Menit)

### Step 1: Install Python Packages
```bash
pip install -r requirements.txt
```

### Step 2: Test MT5 Connection
```bash
python -c "import MetaTrader5 as mt5; print('✓ MT5 OK' if mt5.initialize() else '❌ MT5 Error'); mt5.shutdown()"
```

### Step 3: Launch GUI
```bash
launch_gui.bat
```

## Trading Pertama Kali

### 1. Konfigurasi Dasar
- Symbol: EURUSD (atau pair favorit Anda)
- Volume: 0.01 (micro lot untuk testing)
- Magic Number: 2026001 (unique ID)

### 2. Risk Settings
- Max Daily Loss: $100-500 (sesuai comfort level)
- Max Daily Trades: 100-500
- Risk per Trade: 1% (conservative)

### 3. Start Trading
1. Click "▶️ START TRADING"
2. Monitor tab "Performance"
3. Check tab "Risk Management"

## Tips Pemula

### ✅ DO's
- ✅ Test di DEMO account dulu minimal 1 minggu
- ✅ Start dengan volume kecil (0.01)
- ✅ Monitor performance setiap hari
- ✅ Set realistic risk limits
- ✅ Save configuration setelah setting

### ❌ DON'T's
- ❌ Langsung trade di live account
- ❌ Set volume terlalu besar
- ❌ Abaikan risk management
- ❌ Trade tanpa monitoring
- ❌ Lupa save configuration

## Telegram Setup (Optional)

### 1. Buat Bot
1. Open Telegram
2. Search @BotFather
3. Send: `/newbot`
4. Follow instructions
5. Copy bot token

### 2. Get User ID
1. Search @userinfobot
2. Send: `/start`
3. Copy your ID

### 3. Configure
Edit `telegram_config.json`:
```json
{
    "bot_token": "YOUR_TOKEN_HERE",
    "allowed_users": [YOUR_ID_HERE]
}
```

### 4. Launch Bot
```bash
launch_telegram.bat
```

## ML Training (Optional)

### Quick Train
```bash
launch_train.bat
```

Ini akan:
- Download 30 hari historical data
- Train 2 ML models
- Save models ke folder ./models
- Enable ML predictions

### Manual Train
```python
from ml_predictor import MLPredictor
import MetaTrader5 as mt5

mt5.initialize()
predictor = MLPredictor("EURUSD", {})
predictor.train(days=30)
predictor.save_models("./models")
mt5.shutdown()
```

## Monitoring

### GUI Monitoring
- Tab "Performance": Real-time metrics
- Tab "Risk Management": Risk status
- Tab "Logs": System logs

### Telegram Monitoring
- `/status` - Quick status check
- `/stats` - Trading statistics
- `/performance` - Performance metrics

## Troubleshooting Cepat

### Problem: MT5 not connecting
**Solution:**
1. Pastikan MT5 running
2. Login ke account
3. Restart GUI

### Problem: No signals generated
**Solution:**
1. Check spread (jangan terlalu lebar)
2. Lower min_signal_strength (dari 0.6 ke 0.5)
3. Check market volatility

### Problem: High latency
**Solution:**
1. Close unnecessary programs
2. Check internet connection
3. Consider using VPS

## Next Steps

Setelah comfortable dengan basics:

1. **Optimize Settings**
   - Adjust signal parameters
   - Fine-tune risk settings
   - Test different symbols

2. **Enable ML**
   - Train models
   - Enable ML predictions
   - Compare performance

3. **Scale Up**
   - Increase volume gradually
   - Add more symbols
   - Optimize for your trading style

---

**Need Help?** Check README.md untuk detailed documentation!

**Happy Trading! 🚀**
