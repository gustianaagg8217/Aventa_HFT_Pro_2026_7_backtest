# 🚀 Aventa HFT Pro 2026

**Ultra Low Latency High-Frequency Trading System for MetaTrader 5**

---

## ✨ Fitur Utama

### 🎯 Core Features
- **Ultra-Low Latency Engine** - Tick processing dalam microseconds
- **Advanced Order Flow Analysis** - Analisis mendalam order flow dan volume profile
- **Market Microstructure Analysis** - Spread analysis, price velocity, momentum detection
- **Machine Learning Integration** - Random Forest & Gradient Boosting untuk prediksi
- **Professional Risk Management** - Kelly Criterion, circuit breakers, dynamic position sizing
- **Real-time Monitoring** - GUI modern dengan monitoring real-time
- **Telegram Bot** - Remote monitoring dan control via Telegram

### ⚡ Performance
- **Tick Latency**: < 100 microseconds (μs)
- **Execution Time**: < 10 milliseconds (ms)
- **Order Filling**: IOC (Immediate or Cancel) untuk HFT
- **Multi-threaded**: Data collection, analysis, dan execution dalam thread terpisah

### 🛡️ Risk Management
- Dynamic position sizing dengan Kelly Criterion
- Circuit breaker system untuk emergency stop
- Daily loss limits dan trade count limits
- Maximum drawdown protection
- Real-time risk monitoring
- Trailing stop loss automation

### 🤖 Machine Learning
- Feature engineering untuk 50+ technical indicators
- Random Forest untuk direction prediction
- Gradient Boosting untuk confidence scoring
- Model training dengan historical data
- Real-time prediction integration

---

## 📋 Requirements

### Software
- **Windows 10/11**
- **Python 3.8+**
- **MetaTrader 5**
- **Broker dengan ultra-low latency** (disarankan)

### Python Libraries
```bash
pip install MetaTrader5
pip install numpy pandas
pip install scikit-learn
pip install python-telegram-bot
pip install joblib
```

---

## 🚀 Quick Start

### 1. Instalasi

```bash
# Clone atau download project
cd D:\AVENTA\Aventa_AI\Aventa_HFT_Pro_2026

# Install dependencies
pip install -r requirements.txt
```

### 2. Konfigurasi

Edit `hft_pro_config.json`:
```json
{
    "symbol": "EURUSD",
    "default_volume": 0.01,
    "magic_number": 2026001,
    "risk_per_trade": 0.01,
    "max_daily_loss": 1000,
    "max_daily_trades": 500
}
```

### 3. Train ML Models (Optional)

```bash
# Jalankan training
launch_train.bat

# Atau manual:
python ml_predictor.py
```

### 4. Launch GUI

```bash
# Jalankan GUI launcher
launch_gui.bat

# Atau manual:
python gui_launcher.py
```

### 5. Setup Telegram Bot (Optional)

1. Buat bot baru via @BotFather di Telegram
2. Copy bot token
3. Rename `telegram_config_example.json` ke `telegram_config.json`
4. Edit file dengan bot token dan user ID Anda
5. Jalankan: `launch_telegram.bat`

---

## 📊 Cara Penggunaan

### Menggunakan GUI

1. **Buka GUI Launcher**
   - Double-click `launch_gui.bat`
   - Atau jalankan `python gui_launcher.py`

2. **Konfigurasi Trading**
   - Tab "Control Panel": Set symbol, volume, risk parameters
   - Tab "Risk Management": Set risk limits
   - Tab "ML Models": Train atau load ML models

3. **Start Trading**
   - Klik tombol "▶️ START TRADING"
   - Monitor performance di tab "Performance"
   - Check risk status di tab "Risk Management"

4. **Stop Trading**
   - Klik tombol "⏹️ STOP TRADING"
   - Semua posisi akan ditutup otomatis

### Menggunakan Telegram Bot

1. **Start Bot**
   ```bash
   launch_telegram.bat
   ```

2. **Commands**
   - `/start` - Mulai bot dan lihat menu
   - `/status` - Status sistem dan MT5
   - `/stats` - Statistik trading
   - `/performance` - Metrics performance
   - `/risk` - Status risk management
   - `/positions` - Posisi yang terbuka
   - `/start_trading` - Start trading
   - `/stop_trading` - Stop trading
   - `/close_all` - Close semua posisi

---

## 🔧 Konfigurasi Advanced

### Signal Generation

```json
{
    "min_delta_threshold": 50,
    "min_velocity_threshold": 0.00001,
    "max_spread": 0.0002,
    "min_signal_strength": 0.6,
    "analysis_interval": 0.1
}
```

- `min_delta_threshold`: Minimum cumulative delta untuk signal
- `min_velocity_threshold`: Minimum price velocity untuk momentum
- `max_spread`: Maximum spread yang diperbolehkan
- `min_signal_strength`: Minimum signal confidence (0-1)
- `analysis_interval`: Interval analisis dalam detik

### Risk Management

```json
{
    "risk_per_trade": 0.01,
    "max_daily_loss": 1000,
    "max_daily_trades": 500,
    "max_position_size": 1.0,
    "max_positions": 3,
    "max_drawdown_pct": 10,
    "use_kelly_criterion": true
}
```

- `risk_per_trade`: Risk per trade sebagai % dari balance
- `max_daily_loss`: Maximum loss per hari ($)
- `max_daily_trades`: Maximum jumlah trade per hari
- `max_position_size`: Maximum volume per posisi
- `max_positions`: Maximum jumlah posisi concurrent
- `max_drawdown_pct`: Maximum drawdown (%)
- `use_kelly_criterion`: Gunakan Kelly untuk position sizing

### Position Management

```json
{
    "risk_reward_ratio": 2.0,
    "sl_multiplier": 1.0,
    "use_trailing_stop": true,
    "trail_start_pct": 0.5,
    "trail_distance_pct": 0.3
}
```

- `risk_reward_ratio`: Risk-reward ratio (2.0 = 1:2)
- `sl_multiplier`: Stop loss multiplier
- `use_trailing_stop`: Enable trailing stop
- `trail_start_pct`: Start trailing pada profit % ini
- `trail_distance_pct`: Distance trailing stop dari price

---

## 📈 Performance Optimization

### 1. Tick Latency Optimization
- Gunakan broker dengan ultra-low latency
- Gunakan VPS dekat dengan server broker
- Minimize aplikasi lain yang berjalan

### 2. Execution Optimization
- Set `slippage` yang tepat (default: 20 points)
- Gunakan IOC (Immediate or Cancel) filling mode
- Monitor execution times di tab Performance

### 3. Signal Quality
- Adjust `min_signal_strength` berdasarkan market conditions
- Monitor win rate dan adjust parameters
- Use ML predictions untuk signal enhancement

---

## 🛡️ Safety Features

### Circuit Breakers
System akan otomatis stop trading jika:
- Daily loss limit tercapai
- Maximum drawdown exceeded
- Terlalu banyak consecutive losses

### Risk Controls
- Real-time position monitoring
- Automatic position sizing
- Dynamic stop loss placement
- Trailing stop management

### Monitoring
- Real-time performance metrics
- Risk level indicators
- Trading statistics
- Telegram alerts

---

## 📝 Trading Strategy

### Signal Generation Logic

1. **Order Flow Analysis**
   - Monitor buy/sell volume imbalance
   - Track cumulative delta
   - Detect aggressive buying/selling

2. **Momentum Detection**
   - Calculate price velocity
   - Identify micro-trends
   - Measure acceleration

3. **Spread Conditions**
   - Only trade during tight spreads
   - Filter high volatility periods
   - Wait for optimal entry conditions

4. **Signal Confirmation**
   - Combine multiple indicators
   - Weight by signal strength
   - ML prediction enhancement (optional)

### Position Management

1. **Entry**
   - Market orders with IOC filling
   - Dynamic position sizing
   - Immediate SL/TP placement

2. **Management**
   - Trailing stop activation
   - Partial profit taking (optional)
   - Dynamic SL adjustment

3. **Exit**
   - Hit TP/SL
   - Signal reversal
   - End of day close
   - Manual close

---

## 📊 Monitoring & Analytics

### Real-time Metrics
- Tick processing latency
- Order execution time
- Signal generation count
- Position status
- Daily PnL

### Performance Statistics
- Win rate
- Profit factor
- Average win/loss
- Sharpe ratio
- Maximum drawdown

### Risk Metrics
- Current exposure
- Daily trades used
- Remaining risk capacity
- Drawdown percentage
- Circuit breaker status

---

## 🔍 Troubleshooting

### MT5 Connection Issues
```python
# Check MT5 connection
import MetaTrader5 as mt5
if not mt5.initialize():
    print("Error:", mt5.last_error())
```

### High Latency
- Check internet connection
- Use VPS near broker server
- Close unnecessary applications
- Check broker server status

### No Signals Generated
- Check spread conditions
- Verify configuration parameters
- Check market volatility
- Review min_signal_strength

### Circuit Breaker Triggered
- Review risk limits in config
- Check trading performance
- Wait for next trading day
- Adjust risk parameters

---

## 🎓 Best Practices

### 1. Testing
- **ALWAYS test on demo account first**
- Monitor performance for at least 1 week
- Verify all features working correctly
- Check risk management triggers

### 2. Configuration
- Start with conservative settings
- Gradually increase position size
- Monitor win rate and adjust
- Keep detailed trading logs

### 3. Risk Management
- Never risk more than 1-2% per trade
- Set realistic daily loss limits
- Use circuit breakers
- Monitor drawdown regularly

### 4. Maintenance
- Regular performance review
- Update ML models monthly
- Check for broker updates
- Monitor system resources

---

## 📞 Support & Contact

### Issues & Bugs
Jika menemukan bugs atau issues, please document:
- Error message
- Steps to reproduce
- System configuration
- MT5 version

### Feature Requests
Suggestions untuk improvements are welcome!

---

## ⚠️ Disclaimer

**IMPORTANT:** Trading forex dan CFD melibatkan risiko tinggi dan mungkin tidak cocok untuk semua investor. Anda bisa kehilangan lebih dari investasi awal Anda. Pastikan Anda memahami risiko yang terlibat.

**Penggunaan software ini sepenuhnya adalah tanggung jawab Anda sendiri. Developer tidak bertanggung jawab atas kerugian trading yang mungkin terjadi.**

**ALWAYS test on demo account first!**

---

## 📜 Version History

### v1.0.0 (December 2025)
- ✅ Ultra-low latency core engine
- ✅ Advanced order flow analysis
- ✅ ML prediction models
- ✅ Professional risk management
- ✅ Modern GUI interface
- ✅ Telegram bot integration
- ✅ Comprehensive documentation

---

## 🎯 Roadmap 2026

- [ ] Multi-symbol trading support
- [ ] Advanced chart integration
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Sentiment analysis integration
- [ ] Cloud deployment support
- [ ] Mobile app companion
- [ ] Advanced backtesting framework
- [ ] Real-time dashboard web interface

---

## 🙏 Credits

**Developed by:** Aventa AI Team
**Year:** 2025-2026
**Version:** 1.0.0

Built with ❤️ for professional traders

---

**Happy Trading! 🚀📈**
