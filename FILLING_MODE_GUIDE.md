# Order Filling Mode Guide

## Overview
Different brokers support different order filling policies. The bot now allows you to select the filling mode from the GUI to ensure compatibility with your broker.

## Available Filling Modes

### 1. FOK (Fill or Kill) - **RECOMMENDED FOR HFT**
- **Description**: Order must be filled completely at the specified price or canceled immediately
- **Use Case**: Best for high-frequency trading where partial fills are not acceptable
- **Pros**:
  - Guaranteed full volume or nothing
  - No partial position management needed
  - Predictable execution
- **Cons**:
  - May reject orders during low liquidity
  - Strict execution requirements
- **Broker Support**: Most ECN/STP brokers

### 2. IOC (Immediate or Cancel)
- **Description**: Fill any available volume immediately, cancel the rest
- **Use Case**: When you want to capture whatever liquidity is available
- **Pros**:
  - Higher fill rate than FOK
  - Good for low liquidity instruments
  - Flexible execution
- **Cons**:
  - Partial fills require position tracking
  - May get unexpected volumes
- **Broker Support**: Most professional brokers

### 3. RETURN (Market Execution)
- **Description**: Broker returns the best available price (market order)
- **Use Case**: Market makers, or when price certainty is not critical
- **Pros**:
  - Highest fill probability
  - Works with most brokers
  - Good for market makers
- **Cons**:
  - Price slippage possible
  - Less control over execution price
- **Broker Support**: All brokers, especially market makers

## How to Select Filling Mode

### From GUI:
1. Open GUI Launcher
2. In the "Bot Configuration" tab, find "Filling Mode" dropdown
3. Select your desired mode: **FOK**, **IOC**, or **RETURN**
4. Click "Save Config" to persist your choice
5. Start trading

### From Config File:
```json
{
  "execution_settings": {
    "filling_mode": "FOK",
    "slippage": 50,
    "analysis_interval": 0.5
  }
}
```

## Testing Recommendations

### Step 1: Test with FOK
```
Symbol: GOLD.ls
Filling Mode: FOK
Volume: 0.01
Expected: Clean fills or complete rejections
```

### Step 2: Try IOC if FOK fails
```
Symbol: GOLD.ls
Filling Mode: IOC
Volume: 0.01
Expected: Partial fills allowed
```

### Step 3: Use RETURN as last resort
```
Symbol: GOLD.ls
Filling Mode: RETURN
Volume: 0.01
Expected: Always fills, possible slippage
```

## Broker-Specific Notes

### ECN Brokers (e.g., XM, IC Markets):
- **Best Mode**: FOK or IOC
- **Reason**: True market execution, tight spreads
- **Tip**: FOK works best during high liquidity hours

### Market Maker Brokers:
- **Best Mode**: RETURN
- **Reason**: They provide liquidity, FOK/IOC may be rejected
- **Tip**: Watch for requotes

### Hybrid Brokers:
- **Best Mode**: IOC
- **Reason**: Balance between fill rate and execution quality
- **Tip**: Test during different market hours

## Troubleshooting

### Error: "Order rejected - Invalid filling mode"
**Solution**: Your broker doesn't support the selected mode
1. Try switching to IOC
2. If still fails, use RETURN
3. Check broker specifications

### Error: "Order partially filled" (with FOK)
**Solution**: This shouldn't happen with FOK
1. Verify mode is actually FOK in logs
2. Check if broker changed the request
3. Contact broker support

### Low fill rate with FOK
**Solution**: Not enough liquidity at exact price
1. Switch to IOC for partial fills
2. Increase slippage tolerance
3. Trade during high-volume hours

## Logs to Watch

When you start trading, check for this line:
```
✓ MT5 initialized successfully for GOLD.ls
  Filling mode: FOK
```

During order execution, watch for:
```
✓ Position opened: BUY 0.01 @ 2650.12
```

Or rejection messages:
```
Order failed: 10030 - Invalid filling mode
```

## Best Practices

1. **Start with FOK**: Most reliable for HFT
2. **Test each mode**: Spend 15-30 minutes per mode to collect data
3. **Log everything**: Check logs for any filling mode errors
4. **Match broker type**: ECN=FOK, MM=RETURN
5. **Document results**: Note which mode works best for each symbol/broker

## Default Settings

The bot defaults to **FOK** if no filling mode is specified:
- GOLD.ls: FOK
- EURUSD: FOK
- XAUUSD: FOK

You can override this in GUI or config file.

## Performance Impact

| Mode   | Latency | Fill Rate | Slippage Risk | Best For     |
|--------|---------|-----------|---------------|--------------|
| FOK    | Low     | Medium    | None          | HFT, Scalping|
| IOC    | Low     | High      | Minimal       | Swing, HFT   |
| RETURN | Lowest  | Highest   | Moderate      | Market Making|

## Support

If you encounter filling mode errors:
1. Check broker specifications
2. Test all three modes
3. Review execution logs
4. Contact broker if persistent issues

---
*Last Updated: December 2025*
*Part of Aventa HFT Pro 2026*
