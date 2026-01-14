# ML Model Organization - Per Symbol

## 🎯 Fitur Baru: Model Organization by Symbol

Model ML sekarang disimpan dalam folder terpisah untuk setiap symbol!

---

## 📁 Struktur Folder Baru

```
models/
├── GOLD.ls/
│   ├── lstm_model.h5
│   ├── gru_model.h5
│   ├── cnn_model.h5
│   ├── direction_model.pkl
│   ├── confidence_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── EURUSD/
│   ├── lstm_model.h5
│   ├── gru_model.h5
│   └── ...
│
├── XAUUSD/
│   ├── lstm_model.h5
│   └── ...
│
└── BTCUSD/
    ├── lstm_model.h5
    └── ...
```

---

## ✨ Keuntungan Sistem Baru

### 1. **Organized by Symbol**
- Setiap symbol punya folder sendiri
- Tidak tercampur antara model symbol berbeda
- Mudah manage dan backup per symbol

### 2. **Multiple Symbol Support**
- Train model untuk banyak symbol
- Switch antar symbol dengan mudah
- Setiap symbol punya karakteristik sendiri

### 3. **Easy Selection**
- Dialog selection saat load model
- List semua symbol yang punya model
- Pilih symbol mana yang mau digunakan

---

## 🚀 Cara Menggunakan

### 1. Training Model untuk Symbol Tertentu

```
1. Pilih Symbol di Control Panel (misal: GOLD.ls)
2. Set Training Days (30-90 hari)
3. Klik "🎓 Train Models"
4. Wait untuk training selesai
5. Model otomatis tersimpan ke: models/GOLD.ls/
```

**Lokasi penyimpanan otomatis:** `models/{SYMBOL}/`

### 2. Training Multiple Symbols

```
Training GOLD:
1. Set Symbol = GOLD.ls
2. Train → Saved to models/GOLD.ls/

Training EURUSD:
1. Set Symbol = EURUSD
2. Train → Saved to models/EURUSD/

Training XAUUSD:
1. Set Symbol = XAUUSD
2. Train → Saved to models/XAUUSD/
```

Setiap symbol tersimpan terpisah!

### 3. Load Model dengan Dialog Selection

```
1. Klik "📁 Load Models" di ML Tab
2. Dialog akan muncul dengan list symbol available
3. Pilih symbol yang mau diload:
   
   ┌─────────────────────────────┐
   │  Select Model to Load       │
   ├─────────────────────────────┤
   │  ○ BTCUSD                   │
   │  ○ EURUSD                   │
   │  ✓ GOLD.ls    ← Current     │
   │  ○ XAUUSD                   │
   └─────────────────────────────┘
   
4. Klik "Load"
5. Model loaded dan symbol di GUI berubah otomatis
```

### 4. Save Model (Manual)

```
1. Load atau Train model
2. Klik "💾 Save Models"
3. Model disimpan ke: models/{CURRENT_SYMBOL}/
```

Model otomatis disimpan saat training selesai, tapi bisa manual save juga.

---

## 📊 ML Status Display

Tab ML Models sekarang menampilkan:

### Current Symbol Info:
```
📈 TRAINING INFORMATION:
────────────────────────────────────────────
Current Symbol: GOLD.ls
Model Location: ./models/GOLD.ls/
Available Models: LSTM, GRU, CNN, Direction, Confidence
Last Trained: 2025-12-24 15:30:45
Training Days: 60
Status: ✓ Ready for predictions
```

### All Available Models:
```
📁 ALL AVAILABLE MODELS:
────────────────────────────────────────────
○ BTCUSD
○ EURUSD
✓ GOLD.ls    ← Currently loaded
○ XAUUSD

Use '📁 Load Models' to switch between symbols
```

---

## 🔄 Workflow Example

### Scenario: Trading Multiple Symbols

**Morning - Train GOLD:**
```
1. Symbol = GOLD.ls
2. Train Models (30 days)
3. Models saved to: models/GOLD.ls/
4. Start Trading GOLD
```

**Afternoon - Train EURUSD:**
```
1. Symbol = EURUSD
2. Train Models (60 days)
3. Models saved to: models/EURUSD/
4. Both GOLD and EURUSD models tersimpan!
```

**Evening - Switch to EURUSD:**
```
1. Stop GOLD trading
2. Load Models → Select EURUSD
3. Start Trading EURUSD with its specific model
```

---

## 💡 Best Practices

### 1. **Train Per Symbol**
Setiap symbol punya karakteristik berbeda:
- GOLD: Volatile, range besar
- EURUSD: Smooth, range kecil
- BTCUSD: Sangat volatile

Train terpisah untuk hasil optimal!

### 2. **Regular Retraining**
```
- Weekly: Quick retrain (7-14 days)
- Monthly: Full retrain (30-90 days)
- After major market events: Immediate retrain
```

### 3. **Backup Models**
```
Backup folder models/ secara berkala:
- Copy models/ ke backup location
- Cloud backup (Google Drive, Dropbox)
- External HDD/USB
```

### 4. **Model Naming**
Gunakan symbol exact sesuai broker:
```
✅ GOLD.ls (Instaforex)
✅ EURUSD
✅ XAUUSD (XM)
✅ BTCUSD

❌ Gold
❌ gold
❌ GOLD (kalau broker pakai GOLD.ls)
```

---

## 🛠️ Technical Details

### Model Files per Symbol:
```
{SYMBOL}/
├── lstm_model.h5           # LSTM neural network
├── gru_model.h5            # GRU neural network
├── cnn_model.h5            # CNN neural network
├── direction_model.pkl     # Direction classifier
├── confidence_model.pkl    # Confidence predictor
├── scaler.pkl             # Feature scaler
└── feature_columns.pkl    # Feature names
```

### Auto-Detection:
```python
# System checks these files to detect valid models:
- lstm_model.h5 OR
- gru_model.h5 OR
- cnn_model.h5 OR
- direction_model.pkl OR
- confidence_model.pkl

If any exists → Symbol shown in Load dialog
```

---

## 🎯 Migration from Old System

### If you have old models in `./models/`:
```
Old structure:
models/
├── lstm_model.h5
├── gru_model.h5
└── ...

Move to symbol folder:
models/
└── GOLD.ls/        ← Create this folder
    ├── lstm_model.h5
    ├── gru_model.h5
    └── ...
```

### Auto-Migration:
Training new model will automatically:
1. Create symbol folder
2. Save to correct location
3. Old models tidak terpakai (bisa dihapus)

---

## 📝 Notes

### Symbol Selection Dialog:
- **Sorted alphabetically** for easy finding
- **Current symbol highlighted** with ✓
- **Auto-scroll** to current symbol
- **Double-click** to load (coming soon)

### Auto Symbol Update:
- Load model → Symbol di GUI berubah otomatis
- Konsistensi antara model dan symbol
- No mismatch symbol vs model

### Error Handling:
- No models folder? → Show error + guide
- Empty models folder? → Suggest train first
- Invalid model files? → Clear error message

---

## 🚀 Future Enhancements (Ideas)

- [ ] Model comparison tool
- [ ] Performance metrics per symbol
- [ ] Auto-cleanup old models
- [ ] Model version history
- [ ] Export/Import model sets
- [ ] Cloud sync models
- [ ] Model performance dashboard

---

## 🎉 Summary

**Sekarang lebih organized, flexible, dan powerful!**

✅ One folder per symbol
✅ Easy to manage multiple symbols
✅ Smart load dialog with selection
✅ Auto symbol switching
✅ Clear status display

**Happy Trading with Multiple Symbols! 🚀**
