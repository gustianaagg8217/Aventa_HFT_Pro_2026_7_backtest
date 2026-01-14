# GOLD.ls Configuration Guide - Aventa HFT Pro 2026

## 📊 GOLD.ls Specification (From MT5)

Based on your MT5 screenshot:

```
Symbol:          GOLD.ls, Spot Gold
Digits:          2 (price format: 2650.12)
Contract size:   100
Spread:          floating (typically 2.00 points)
Stops level:     80
Tick size:       0.01
Tick value:      1
Hedged margin:   50
Calculation:     CFD
```

---

## 🎯 **CRITICAL: Correct Spread Calculation**

### ❌ **WRONG** (Previous assumption):
```python
Spread = 2 points
Point = 0.001 (like forex)
Actual spread = 2 × 0.001 = 0.002
max_spread = 0.003  # TOO SMALL! ❌
```

### ✅ **CORRECT** (Based on MT5 spec):
```python
Spread = 2 points
Tick size = 0.01 (from specification)
Actual spread = 2 × 0.01 = 0.02
max_spread = 0.05  # CORRECT! ✅
```

**GOLD memiliki tick size 10x lebih besar dari forex!**

---

## ✅ **Recommended Configuration for GOLD.ls**

### **Quick Start - Use Preset Config:**

```bash
# Load preset config for GOLD
python gui_launcher.py
# File → Load Config → config_GOLD.json
```

### **Or Manual Settings in GUI:**

```
┌─────────────────────────────────────────────────┐
│ Configuration for GOLD.ls                        │
├─────────────────────────────────────────────────┤
│ Symbol:          GOLD.ls                         │
│ Default Volume:  0.01                            │
│ Magic Number:    2026001                         │
│ Risk per Trade:  1.0%                            │
│ Min Signal Str:  0.4 (0.3 for more trades)      │
│ Max Spread:      0.05  ⭐ IMPORTANT!             │
│ Max Volatility:  0.1   ⭐ IMPORTANT!             │
└─────────────────────────────────────────────────┘
```

### **Key Settings Explained:**

#### 1. **Max Spread: 0.05** ⭐
```
Why: GOLD spread = 2pt × 0.01 = 0.02
Set to 0.05 to allow some spread fluctuation
Range: 0.02 - 0.1 (0.02 = tight, 0.1 = loose)
```

#### 2. **Max Volatility: 0.1** ⭐
```
Why: GOLD is much more volatile than forex
EURUSD typical: 0.001
GOLD typical: 0.01 - 0.1
```

#### 3. **Min Signal Strength: 0.4**
```
0.3 = More aggressive (more trades)
0.4 = Balanced (recommended)
0.5-0.6 = Conservative (fewer, safer trades)
```

#### 4. **Min Velocity Threshold: 0.0001**
```
Why: GOLD price moves in larger increments (0.01)
EURUSD uses: 0.00001
GOLD uses: 0.0001 (10x larger)
```

#### 5. **Min Delta Threshold: 20**
```
Order flow analysis threshold
Lower = more sensitive
Range: 10-50 for testing, 50-100 for live
```

---

## 📋 **Complete Config File (config_GOLD.json)**

```json
{
  "symbol": "GOLD.ls",
  "magic_number": 2026001,
  "default_volume": 0.01,
  
  "max_spread": 0.05,
  "max_volatility": 0.1,
  "min_signal_strength": 0.4,
  "min_delta_threshold": 20,
  "min_velocity_threshold": 0.0001,
  "risk_reward_ratio": 2.0,
  "analysis_interval": 0.5,
  "slippage": 50,
  
  "risk_per_trade": 0.01,
  "max_daily_loss": 1000,
  "max_daily_trades": 100,
  "max_position_size": 1.0,
  "max_positions": 3,
  "max_drawdown_pct": 10
}
```

---

## 🚀 **Quick Test with Correct Settings**

```bash
python quick_test.py
# Enter: GOLD.ls
```

**Updated quick_test.py now uses:**
- ✅ max_spread = 0.05 (correct for GOLD!)
- ✅ max_volatility = 0.1 (adjusted for GOLD)
- ✅ min_velocity_threshold = 0.0001 (larger movements)
- ✅ min_signal_strength = 0.3 (aggressive for testing)

---

## 📊 **Expected Bot Behavior**

### **Startup Logs (Should Look Like):**

```
2025-12-24 15:00:00.000 - INFO - ============================================================
2025-12-24 15:00:00.000 - INFO - Starting Aventa HFT Pro 2026 Engine
2025-12-24 15:00:00.000 - INFO - ============================================================
2025-12-24 15:00:00.100 - INFO - ✓ MT5 initialized successfully for GOLD.ls
2025-12-24 15:00:00.100 - INFO -   Spread: 2.00 points
2025-12-24 15:00:00.100 - INFO -   Tick size: 0.01
2025-12-24 15:00:00.100 - INFO -   Tick value: 1.0
2025-12-24 15:00:00.100 - INFO -   Point: 0.01
2025-12-24 15:00:00.100 - INFO -   Spread (price): 0.02000
2025-12-24 15:00:00.100 - INFO -   Max Spread Setting: 0.05000
2025-12-24 15:00:00.100 - INFO -   ✓ Spread OK for trading  ⭐
2025-12-24 15:00:00.200 - INFO - ✓ All threads started successfully
2025-12-24 15:00:00.200 - INFO -   Symbol: GOLD.ls
2025-12-24 15:00:00.200 - INFO -   Analysis interval: 0.5s
2025-12-24 15:00:00.200 - INFO -   Default volume: 0.01
2025-12-24 15:00:00.200 - INFO -   Min signal strength: 0.4
2025-12-24 15:00:00.200 - INFO -   Risk/Reward ratio: 2.0
2025-12-24 15:00:00.200 - INFO - 
2025-12-24 15:00:00.200 - INFO - 🔍 Waiting for trading signals...
```

### **During Operation:**

```
2025-12-24 15:00:10.000 - INFO - ⏳ Analyzing market... (50 analyses, no strong signal yet)
2025-12-24 15:00:20.000 - INFO - ⏳ Analyzing market... (100 analyses, no strong signal yet)
2025-12-24 15:00:35.500 - INFO - 📊 SIGNAL GENERATED: BUY | Strength: 0.65 | Price: 2650.25 | Reason: Positive delta: 25 | Positive momentum: 0.00015
2025-12-24 15:00:35.550 - INFO - 📈 Attempting to open BUY position...
2025-12-24 15:00:35.650 - INFO - ✓ Executed BUY | Price: 2650.25 | Strength: 0.65 | Time: 100.00ms
```

---

## ⚠️ **Common Issues & Solutions**

### Issue 1: "Spread too wide" warnings

**Symptoms:**
```
⚠️ Spread too wide: 0.02500 > 0.00300 (max)
```

**Solution:**
```python
# Increase max_spread in GUI or config
max_spread = 0.05  # or higher
```

### Issue 2: No signals generated

**Symptoms:**
```
⏳ Analyzing market... (500 analyses, no strong signal yet)
(No weak signal logs either)
```

**Possible causes:**
- Market not active (check time)
- Thresholds too high
- Not enough tick data yet (wait 30-60 seconds)

**Solution:**
```python
# Lower thresholds temporarily for testing:
min_signal_strength = 0.3
min_delta_threshold = 10
min_velocity_threshold = 0.00005
```

### Issue 3: Weak signals but no execution

**Symptoms:**
```
⚠️ Weak signal: BUY | Strength: 0.45 < 0.60 (threshold)
```

**Solution:**
```python
# Lower min_signal_strength
min_signal_strength = 0.4  # or 0.3
```

### Issue 4: MT5 symbol not found

**Symptoms:**
```
ERROR - Symbol GOLD.ls not found
```

**Solution:**
```python
# Try alternative gold symbols:
"XAUUSD"  # Common alternative
"GOLD"    # Some brokers
"Gold"    # Case sensitive

# Check in MT5 Market Watch for exact symbol name
```

---

## 🎯 **Performance Targets for GOLD.ls**

With correct settings, you should see:

### **Tick Processing:**
- ✅ Tick latency: < 500μs (0.5ms)
- ✅ Execution time: < 200ms
- ✅ Ticks processed: Increasing steadily
- ✅ Order flow samples: Building up

### **Signal Generation:**
- ✅ Signals per hour: 5-20 (depends on market activity)
- ✅ Signal strength: Average 0.5-0.7
- ✅ Win rate target: > 50%

### **Trading Performance:**
- ✅ Trades per day: 20-100 (depends on settings)
- ✅ Average profit per trade: > spread cost
- ✅ Max drawdown: < 10%

---

## 💡 **Tips for GOLD Trading**

### 1. **Best Trading Hours:**
```
Most volatile (more opportunities):
- London open: 08:00-12:00 GMT
- NY open: 13:00-17:00 GMT
- London/NY overlap: Best liquidity

Less volatile (tighter spreads):
- Asian session: Lower volume
- Weekend: Market closed
```

### 2. **Spread Monitoring:**
```
Check spread throughout the day:
- Typically 2-3 points during active hours
- Can widen to 5-10 points during:
  * News releases
  * Market open/close
  * Low liquidity periods
```

### 3. **Volume Management:**
```
Start small:
- Test with 0.01 lots
- Monitor performance for 1 week
- Gradually increase to 0.05-0.10 lots
- Max recommended: 1.0 lot per position
```

### 4. **Risk Management:**
```
Gold is more volatile:
- Use tighter stop losses (50-100 points)
- Take profit at 2x risk minimum
- Don't over-leverage
- Respect max daily loss limit
```

---

## 📈 **Optimization Process**

### Phase 1: Testing (Week 1)
```python
# Conservative settings
min_signal_strength = 0.5
max_spread = 0.05
max_volatility = 0.1
default_volume = 0.01
```

Monitor:
- Number of trades per day
- Win rate
- Average profit per trade

### Phase 2: Adjustment (Week 2-3)
```python
# If too few trades:
min_signal_strength = 0.4

# If too many losing trades:
min_signal_strength = 0.6

# If spread issues:
max_spread = 0.08
```

### Phase 3: Optimization (Week 4+)
```python
# Fine-tune based on data
# Adjust thresholds per market conditions
# Implement time-based rules
```

---

## ✅ **Checklist Before Live Trading**

- [ ] Tested with quick_test.py
- [ ] Verified spread calculation
- [ ] Set correct max_spread (0.05+)
- [ ] Set correct max_volatility (0.1+)
- [ ] Configured risk limits
- [ ] Monitored for 24 hours on demo
- [ ] Win rate > 50%
- [ ] Max drawdown < 10%
- [ ] Start with 0.01 lot size
- [ ] Have emergency stop plan

---

## 📞 **Support Files**

Created for this setup:
- ✅ [config_GOLD.json](config_GOLD.json) - Preset config
- ✅ [quick_test.py](quick_test.py) - Updated with GOLD settings
- ✅ [gui_launcher.py](gui_launcher.py) - Updated defaults
- ✅ This README

---

## 🚀 **Ready to Test!**

```bash
# Option 1: Quick test (2 minutes)
python quick_test.py

# Option 2: GUI with proper settings
python gui_launcher.py
# Set: Symbol=GOLD.ls, Max Spread=0.05, Max Volatility=0.1
```

**Bot is now configured correctly for GOLD.ls!** 🎉

---

**Last Updated:** December 24, 2025  
**Version:** 2.0 - Corrected for GOLD.ls tick size 0.01
