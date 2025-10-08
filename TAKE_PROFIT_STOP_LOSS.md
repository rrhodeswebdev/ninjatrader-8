# Take Profit & Stop Loss Feature

## Overview

The AITrader strategy now includes configurable Take Profit (TP) and Stop Loss (SL) functionality to protect your trades and lock in profits automatically.

---

## ✨ Features Added

### 1. **Configurable TP/SL Parameters**
- **Take Profit**: Set in ticks (0 = disabled)
- **Stop Loss**: Set in ticks (0 = disabled)
- Both accessible in NinjaTrader strategy parameters

### 2. **Dashboard Display**
- Real-time display of TP/SL settings
- Shows "Disabled" when set to 0
- Shows tick values when enabled

### 3. **Automatic Order Placement**
- TP/SL orders automatically placed on every entry
- Works for both LONG and SHORT trades
- Applies on reversals too

### 4. **Output Logging**
- Logs when TP/SL orders are set
- Shows tick values in output window

---

## 📊 How to Configure

### In NinjaTrader Strategy Parameters:

1. **Open Strategy Parameters:**
   - Right-click chart → Strategies → Configure strategy
   - Or Strategy Analyzer → Strategy settings

2. **Risk Management Section:**
   ```
   Take Profit (Ticks): [0-500]  (Default: 20)
   Stop Loss (Ticks):   [0-500]  (Default: 10)
   ```

3. **Settings:**
   - **0 = Disabled**: No TP/SL orders placed
   - **1-500 ticks**: Places orders at specified distance

---

## 📋 Examples

### Example 1: Conservative Trading
```
Take Profit: 20 ticks
Stop Loss: 10 ticks
Risk/Reward Ratio: 1:2
```

**Output:**
```
EXECUTING LONG ENTRY - Confidence: 72.34%
  → Take Profit set at 20 ticks
  → Stop Loss set at 10 ticks
```

### Example 2: Swing Trading
```
Take Profit: 100 ticks
Stop Loss: 50 ticks
Risk/Reward Ratio: 1:2
```

### Example 3: AI Signals Only (No TP/SL)
```
Take Profit: 0 ticks (Disabled)
Stop Loss: 0 ticks (Disabled)
```

**Output:**
```
EXECUTING LONG ENTRY - Confidence: 72.34%
(No TP/SL messages - relies on AI exit signals)
```

---

## 🎯 Default Settings

**Strategy comes with sensible defaults:**
- **Take Profit:** 20 ticks
- **Stop Loss:** 10 ticks
- **Risk/Reward:** 1:2 ratio

These defaults work well for:
- ES futures (E-mini S&P 500)
- NQ futures (E-mini NASDAQ)
- Most liquid futures contracts

---

## 📈 Dashboard Display

The dashboard now shows:

```
┌─────────────────────────────────────┐
│ AI TRADER DASHBOARD                 │
│                                     │
│ Server: Connected                   │
│ Mode: BOTH                          │
│                                     │
│ LAST SIGNAL                         │
│ Signal: LONG                        │
│ Confidence: 72.34%                  │
│ Time: 14:35:22                      │
│                                     │
│ SIGNAL STATS                        │
│ Total: 145                          │
│ Long: 58 (40.0%)                    │
│ Short: 52 (35.9%)                   │
│ Hold: 35 (24.1%)                    │
│                                     │
│ RISK MANAGEMENT              ← NEW  │
│ Take Profit: 20 ticks        ← NEW  │
│ Stop Loss: 10 ticks          ← NEW  │
└─────────────────────────────────────┘
```

---

## 🔧 Technical Details

### Order Placement

**For LONG Trades:**
```csharp
EnterLong(1, "AILong");
SetProfitTarget("AILong", CalculationMode.Ticks, takeProfitTicks);
SetStopLoss("AILong", CalculationMode.Ticks, stopLossTicks, false);
```

**For SHORT Trades:**
```csharp
EnterShort(1, "AIShort");
SetProfitTarget("AIShort", CalculationMode.Ticks, takeProfitTicks);
SetStopLoss("AIShort", CalculationMode.Ticks, stopLossTicks, false);
```

### Order Naming
- Long entries: "AILong"
- Short entries: "AIShort"
- Allows tracking specific AI trades

### Stop Loss Type
- **Static Stop Loss** (`false` parameter)
- Does not trail
- Fixed distance from entry

---

## 💡 Best Practices

### 1. **Risk/Reward Ratio**
- Maintain at least 1:1.5 ratio
- Example: 10 tick SL → 15+ tick TP
- Default 1:2 is recommended

### 2. **Instrument-Specific Settings**

**ES (E-mini S&P 500):**
```
Scalping:  TP: 8-12 ticks,  SL: 4-6 ticks
Day Trade: TP: 20-30 ticks, SL: 10-15 ticks
Swing:     TP: 50+ ticks,   SL: 25+ ticks
```

**NQ (E-mini NASDAQ):**
```
Scalping:  TP: 10-15 ticks, SL: 5-8 ticks
Day Trade: TP: 25-40 ticks, SL: 12-20 ticks
Swing:     TP: 80+ ticks,   SL: 40+ ticks
```

**Forex (EUR/USD):**
```
Scalping:  TP: 10 pips,  SL: 5 pips
Day Trade: TP: 30 pips,  SL: 15 pips
Swing:     TP: 100 pips, SL: 50 pips
```

### 3. **Confidence-Based Adjustments**
Consider adjusting TP/SL based on AI confidence:
- High confidence (>70%): Wider TP, tighter SL
- Low confidence (<50%): Tighter TP, wider SL
- (This would require custom modification)

### 4. **Time-Based Rules**
- Tighter TP/SL during high volatility (news events)
- Wider TP/SL during low volatility (overnight)
- Adjust based on session (Asia, London, NY)

---

## 📊 Performance Impact

### With TP/SL (Default):
```
✅ Controlled risk per trade
✅ Locked-in profits
✅ Reduced drawdown
✅ Better win rate (smaller targets)
⚠️  May exit early on strong moves
```

### Without TP/SL (AI exits only):
```
✅ Rides trends longer
✅ Larger winners possible
⚠️  Higher drawdown risk
⚠️  Depends on AI exit timing
❌ No protection during gaps
```

---

## 🧪 Backtesting Recommendations

Test different TP/SL combinations:

```
Test Matrix:
─────────────────────────────────
TP (ticks)  │ SL (ticks)  │ R:R
─────────────────────────────────
10          │ 5           │ 1:2
15          │ 10          │ 1:1.5
20          │ 10          │ 1:2  ← Default
30          │ 15          │ 1:2
50          │ 25          │ 1:2
0           │ 0           │ N/A  (AI only)
─────────────────────────────────
```

Compare:
- Win rate
- Profit factor
- Max drawdown
- Sharpe ratio
- Average trade P&L

---

## ⚙️ Advanced Usage

### Disable TP, Keep SL (Risk Protection Only):
```
Take Profit: 0 ticks
Stop Loss: 10 ticks
```
Let AI signals close profitable trades, but protect against losses.

### Disable SL, Keep TP (Profit Taking Only):
```
Take Profit: 20 ticks
Stop Loss: 0 ticks
```
Lock in profits automatically, let AI manage risk.

### Dynamic TP/SL (Future Enhancement):
Could modify strategy to adjust TP/SL based on:
- Market volatility (ATR)
- AI confidence level
- Time of day
- Recent performance

---

## 🐛 Troubleshooting

### TP/SL Not Working?

1. **Check Parameters:**
   - Verify not set to 0 (disabled)
   - Confirm in "Risk Management" section

2. **Check Output Window:**
   - Should see "Take Profit set at X ticks"
   - Should see "Stop Loss set at X ticks"

3. **Check Orders Panel:**
   - Look for PT (Profit Target) orders
   - Look for SL (Stop Loss) orders

4. **Verify Strategy Mode:**
   - Must be in State.Realtime for live trading
   - Historical data won't place actual orders

### Orders Not Filling?

1. **Check Tick Size:**
   - ES: 0.25 points = 1 tick
   - NQ: 0.25 points = 1 tick
   - Ensure TP/SL prices are valid

2. **Check Market Hours:**
   - Some exchanges have restricted hours
   - OCO orders may not work overnight

---

## 📝 Summary

**What's New:**
- ✅ Take Profit parameter (0-500 ticks)
- ✅ Stop Loss parameter (0-500 ticks)
- ✅ Dashboard display of TP/SL settings
- ✅ Automatic order placement on all entries
- ✅ Output logging for transparency

**Default Settings:**
- Take Profit: 20 ticks
- Stop Loss: 10 ticks
- Risk/Reward: 1:2 ratio

**How to Use:**
1. Configure in strategy parameters (Risk Management section)
2. Set to 0 to disable
3. Monitor dashboard for current settings
4. Check output window for order confirmation

**Your AI trading system now has built-in risk management!** 🎯
