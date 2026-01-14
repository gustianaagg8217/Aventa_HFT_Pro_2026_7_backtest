# Spread Settings Guide - Aventa HFT Pro 2026

## 📊 Understanding Spread

**Spread** adalah selisih antara harga Bid dan Ask:
```
Spread = Ask Price - Bid Price
```

Spread yang terlalu lebar akan menghalangi bot untuk trading karena:
- Biaya trading lebih tinggi
- Risk/Reward ratio tidak bagus
- Slippage lebih besar

---

## 🎯 Recommended Max Spread Settings

### **Forex Pairs:**

| Symbol | Typical Spread | Recommended Max Spread | Notes |
|--------|---------------|----------------------|-------|
| EURUSD | 0.1-2 pips | `0.0001 - 0.0003` | Major pair, tight spread |
| GBPUSD | 0.5-2 pips | `0.0002 - 0.0003` | Major pair |
| USDJPY | 0.2-1 pip | `0.002 - 0.01` | JPY pairs use different scale |
| AUDUSD | 0.5-2 pips | `0.0002 - 0.0003` | Major pair |
| EURJPY | 1-3 pips | `0.01 - 0.03` | Cross pair, wider spread |

### **Metals:**

| Symbol | Typical Spread | Recommended Max Spread | Notes |
|--------|---------------|----------------------|-------|
| **GOLD / XAUUSD** | 2-5 points | `0.003 - 0.01` | ⭐ Common issue! |
| GOLD.ls | 2-5 points | `0.003 - 0.01` | Broker-specific symbol |
| SILVER / XAGUSD | 3-10 points | `0.01 - 0.03` | More volatile |

### **Indices:**

| Symbol | Typical Spread | Recommended Max Spread | Notes |
|--------|---------------|----------------------|-------|
| US30 (Dow) | 2-5 points | `1.0 - 5.0` | Large numbers |
| NAS100 | 1-3 points | `0.5 - 3.0` | Tech index |
| GER40 (DAX) | 1-3 points | `0.5 - 3.0` | European index |

---

## 🔧 How to Set Max Spread in GUI

1. **Open GUI Launcher**
   ```bash
   python gui_launcher.py
   ```

2. **Go to Control Panel Tab**

3. **Find "Max Spread" field**
   - Default: `0.003` (good for GOLD)
   - For EURUSD: Use `0.0002`
   - For GOLD: Use `0.003` - `0.01`

4. **Adjust based on your symbol:**
   ```
   GOLD.ls with 2pt spread → Set to 0.003 or higher
   EURUSD with 0.5pip spread → Set to 0.0002
   ```

5. **Click "Start Trading"**

---

## ⚠️ Common Issue: GOLD Spread

### Problem:
```
[2025-12-24] INFO: Symbol: GOLD.ls | Volume: 0.01
(No trades happening)
```

### Diagnosis:
```python
# Check actual spread
import MetaTrader5 as mt5
mt5.initialize()
info = mt5.symbol_info("GOLD.ls")
print(f"Spread: {info.spread} points")  # e.g., 2.0 points
print(f"Point: {info.point}")            # e.g., 0.001
# Actual spread in price = 2.0 * 0.001 = 0.002
```

### Solution:
**Gold typically has 2-5 point spread = 0.002 - 0.005 in price**

Set `Max Spread` to **0.003** or **0.01** for Gold trading!

```python
# In config or GUI:
'max_spread': 0.01  # Allow up to 0.01 spread for Gold
```

---

## 📈 Bot Behavior with Spread

### When Spread is OK:
```
✓ MT5 initialized successfully for GOLD.ls
  Spread: 2.00 points
  Spread (price): 0.00200
  Max Spread Setting: 0.01000
  ✓ Spread OK for trading
```

### When Spread is Too Wide:
```
✓ MT5 initialized successfully for GOLD.ls
  Spread: 2.00 points
  Spread (price): 0.00200
  Max Spread Setting: 0.00050
  ⚠️ Current spread (0.00200) > Max allowed (0.00050)
     Bot may not trade until spread narrows!
     Consider increasing 'Max Spread' setting
```

Bot will:
- ❌ Skip signal generation when spread > max
- ⏸️ Wait for spread to narrow
- 📝 Log occasionally: "Spread too wide"

---

## 🎚️ Dynamic Spread vs Fixed Setting

### Fixed Setting (Current Implementation):
```python
'max_spread': 0.003  # Fixed threshold
```
- Simple
- Predictable
- Works well for most scenarios

### Dynamic Setting (Future):
```python
# Adjust based on symbol automatically
if symbol.startswith("GOLD") or symbol.startswith("XAU"):
    max_spread = 0.01
elif symbol.endswith("JPY"):
    max_spread = 0.03
else:  # Major forex
    max_spread = 0.0003
```

---

## 🧪 Testing Spread Settings

### Test Script:
```python
import MetaTrader5 as mt5

mt5.initialize()

symbols = ["EURUSD", "GOLD.ls", "XAUUSD", "USDJPY"]

for symbol in symbols:
    info = mt5.symbol_info(symbol)
    if info:
        tick = mt5.symbol_info_tick(symbol)
        spread_points = info.spread
        spread_price = spread_points * info.point
        
        print(f"\n{symbol}:")
        print(f"  Spread: {spread_points} points")
        print(f"  Point: {info.point}")
        print(f"  Spread (price): {spread_price:.5f}")
        print(f"  Bid: {tick.bid}")
        print(f"  Ask: {tick.ask}")
        print(f"  Calculated: {tick.ask - tick.bid:.5f}")

mt5.shutdown()
```

### Quick Test with Different Spread:
```bash
python quick_test.py
# Enter: GOLD.ls
# Check logs for spread warnings
```

---

## 📊 Monitoring Spread During Trading

### In GUI Logs:
Look for messages like:
```
⚠️ Spread too wide: 0.00500 > 0.00300 (max)
⏳ Analyzing market... (no strong signal yet)
```

If you see many spread warnings:
1. **Increase Max Spread** setting
2. Or **wait** for tighter spread hours
3. Or **switch to another symbol** with tighter spread

### Peak Spread Hours:
- **Wider spreads:** Market open/close, low liquidity, news events
- **Tighter spreads:** London/NY overlap (8am-12pm EST), high liquidity

---

## 💡 Best Practices

### 1. **Know Your Symbol's Typical Spread**
Before trading, check average spread:
```python
# Monitor for 5 minutes
spreads = []
for i in range(300):  # 5 min at 1sec interval
    tick = mt5.symbol_info_tick("GOLD.ls")
    info = mt5.symbol_info("GOLD.ls")
    spreads.append(info.spread * info.point)
    time.sleep(1)

avg_spread = np.mean(spreads)
max_spread = np.max(spreads)
print(f"Avg: {avg_spread:.5f}, Max: {max_spread:.5f}")
# Set your threshold slightly above average
```

### 2. **Start Conservative, Then Adjust**
```python
# Initial setting (conservative):
'max_spread': 0.001

# If no trades and spread OK in logs:
'max_spread': 0.003

# For Gold specifically:
'max_spread': 0.01
```

### 3. **Different Settings for Different Symbols**
Save configs per symbol:
- `config_eurusd.json` → max_spread: 0.0002
- `config_gold.json` → max_spread: 0.01
- `config_usdjpy.json` → max_spread: 0.03

### 4. **Check Broker Spread Patterns**
- Some brokers widen spread during news
- Some have fixed spreads
- Some have variable spreads
- **Monitor your broker's typical spread before live trading!**

---

## 🔍 Troubleshooting

### Problem: Bot not trading despite good signals

#### Check 1: Spread Setting
```python
# In logs, look for:
"Max Spread Setting: 0.00050"
"Spread (price): 0.00200"
# 0.002 > 0.0005 = TOO TIGHT!
```
**Solution:** Increase max_spread to 0.003+

#### Check 2: Actual Spread
```python
import MetaTrader5 as mt5
mt5.initialize()
info = mt5.symbol_info("GOLD.ls")
print(f"Current spread: {info.spread * info.point}")
```
**Solution:** Compare with your max_spread setting

#### Check 3: Spread Variability
- Spread changes throughout the day
- Check if spread narrows during active hours
- Consider trading only during low-spread hours

---

## 📋 Quick Reference Card

| Your Situation | Recommended Max Spread |
|----------------|----------------------|
| Trading EURUSD, GBPUSD | `0.0002 - 0.0003` |
| Trading GOLD/XAUUSD | `0.003 - 0.01` ⭐ |
| Trading JPY pairs | `0.01 - 0.03` |
| High frequency trading | Lower (tighter control) |
| Swing/Position trading | Higher (less critical) |
| Testing phase | Higher (get more trades) |
| Live trading | Lower (be selective) |

---

## 🚀 Updated Files

After this update, you can now:
- ✅ Set Max Spread in GUI
- ✅ Set Max Volatility in GUI
- ✅ See spread warnings in logs
- ✅ Get spread comparison at startup
- ✅ Understand why bot isn't trading

---

**For GOLD.ls trading, use: max_spread = 0.003 to 0.01**

**Last Updated:** December 24, 2025  
**Version:** 1.1
