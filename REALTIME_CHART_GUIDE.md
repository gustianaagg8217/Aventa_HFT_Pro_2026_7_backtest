# Real-Time Performance Chart - Implementation Guide

## 📊 Overview
Real-time performance chart telah berhasil diimplementasikan di GUI Aventa HFT Pro 2026. Chart ini menampilkan visualisasi performa trading secara langsung dengan update setiap 1 detik.

## ✨ Features

### 3 Chart Panels:

#### 1. **Daily PnL ($)**
- Menampilkan profit/loss kumulatif real-time
- Garis horizontal di angka 0 sebagai referensi
- Warna hijau untuk visualisasi yang jelas
- Auto-scaling untuk menyesuaikan range data

#### 2. **Win Rate & Trade Count**
- **Win Rate (%)**: Ditampilkan pada Y-axis kiri (hijau)
- **Trade Count**: Ditampilkan pada Y-axis kanan (biru)
- Dual-axis chart untuk membandingkan 2 metrik sekaligus
- Win rate dalam persentase (0-100%)

#### 3. **Tick Latency (μs)**
- Menampilkan latency pemrosesan tick dalam microseconds
- Penting untuk monitoring performa HFT
- Warna merah untuk kemudahan identifikasi spike latency

## 🎯 Data Points

Chart menyimpan **300 data points terakhir** (sekitar 5 menit dengan update 1 detik):

```python
self.chart_data = {
    'timestamps': deque(maxlen=300),  # Waktu
    'pnl': deque(maxlen=300),         # Profit/Loss
    'equity': deque(maxlen=300),      # Equity (future use)
    'trades': deque(maxlen=300),      # Jumlah trade
    'win_rate': deque(maxlen=300),    # Win rate percentage
    'latency': deque(maxlen=300)      # Tick latency μs
}
```

## 🚀 Usage

### Starting the Chart:
1. Buka GUI dengan menjalankan `gui_launcher.py`
2. Pergi ke tab **"📊 Performance"**
3. Klik tombol **"🚀 Start Trading"** di tab Control Panel
4. Chart akan otomatis mulai update setiap 1 detik

### Controls:
- **🔄 Reset Chart**: Membersihkan semua data chart dan memulai dari awal
- Chart akan terus update selama bot trading berjalan

## 🔧 Technical Details

### Update Mechanism:
```python
def update_loop(self):
    while self.is_running:
        # Collect performance data
        stats = self.engine.get_performance_stats()
        risk_summary = self.risk_manager.get_trading_summary()
        
        # Store to chart data
        self.chart_data['timestamps'].append(datetime.now())
        self.chart_data['pnl'].append(risk_summary['daily_pnl'])
        self.chart_data['trades'].append(risk_summary['daily_trades'])
        # ... etc
        
        # Update chart visualization
        self.update_chart()
        
        threading.Event().wait(1.0)  # Update every second
```

### Chart Rendering:
- Menggunakan **matplotlib** dengan backend TkAgg
- **FigureCanvasTkAgg** untuk embedding ke tkinter
- **draw_idle()** untuk efficient rendering tanpa blocking UI
- Date formatter untuk menampilkan time dengan format HH:MM:SS

## 📦 Dependencies

Tambahan di `requirements.txt`:
```
matplotlib>=3.5.0
```

Install dengan:
```bash
pip install matplotlib
```

## 🎨 Customization

### Theme (Dark Mode):
```python
self.fig = Figure(figsize=(10, 6), facecolor='#1e1e1e')
ax.set_facecolor('#2d2d2d')
ax.tick_params(colors='#ffffff')
ax.spines['bottom'].set_color('#555555')
```

### Colors:
- **PnL**: `#4ec9b0` (Teal/Green)
- **Win Rate**: `#4ec9b0` (Teal/Green)
- **Trades**: `#007acc` (Blue)
- **Latency**: `#f48771` (Red/Orange)

## 🔍 Data Sources

Chart mengambil data dari:

1. **Engine Stats** (`UltraLowLatencyEngine`):
   - `tick_latency_avg_us`: Average tick processing latency
   - `ticks_processed`: Total ticks processed
   - `current_position`: Current position status

2. **Risk Manager Stats** (`RiskManager`):
   - `daily_pnl`: Total profit/loss for the day
   - `daily_trades`: Number of trades executed today
   - `win_rate`: Percentage of winning trades
   - `equity`: Account equity (planned)

## ⚡ Performance Optimization

- **deque with maxlen**: Automatic old data removal, constant memory
- **draw_idle()**: Non-blocking canvas update
- **relim() + autoscale_view()**: Efficient axis rescaling
- **Threading**: Update loop runs in separate thread

## 🐛 Troubleshooting

### Chart tidak update:
1. Pastikan trading engine sudah di-start
2. Check console untuk error messages
3. Verify risk_manager initialized properly

### Chart lag/freezing:
1. Reduce update frequency (increase sleep time)
2. Reduce maxlen of deque (fewer data points)
3. Close other resource-intensive applications

### Import error matplotlib:
```bash
pip install --upgrade matplotlib
# or
pip install matplotlib --force-reinstall
```

## 📈 Future Enhancements

Planned features:
- [ ] Export chart sebagai PNG/PDF
- [ ] Multiple timeframe views (1min, 5min, 15min, 1hour)
- [ ] Equity curve overlay
- [ ] Drawdown visualization
- [ ] Trade markers pada chart
- [ ] Zoom & Pan controls
- [ ] Custom indicator overlay
- [ ] Performance comparison with benchmark

## ✅ Testing

Gunakan `test_chart.py` untuk testing chart dengan data simulasi:

```bash
python test_chart.py
```

Test script akan:
- Generate random trading data
- Update chart setiap 0.5 detik
- Menampilkan semua 3 chart panels
- Test Start/Stop/Reset functionality

## 📝 Notes

- Chart update rate: **1 second**
- Maximum data points: **300** (5 minutes)
- Memory efficient dengan deque
- Thread-safe update mechanism
- Compatible dengan dark theme GUI

---

**Status**: ✅ **FULLY IMPLEMENTED & TESTED**

**Version**: 1.0  
**Date**: December 24, 2025  
**Author**: Aventa AI Team
