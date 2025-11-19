# AITrader.cs - Contract Auto-Detection Update

## Purpose

Update the NinjaTrader strategy to send the instrument name (MNQ, ES, etc.) to the RNN server for automatic contract detection.

## Changes Required

### 1. Historical Data Section

**Location**: Around line 530-540 in `SendHistoricalData()` method

**Find this line**:
```csharp
jsonBuilder.Append("],\"type\":\"historical\"");
```

**Add immediately after it**:
```csharp
// Add contract information for auto-detection
jsonBuilder.AppendFormat(",\"contract\":\"{0}\"", Instrument.MasterInstrument.Name);
```

**Complete example**:
```csharp
jsonBuilder.Append("],\"type\":\"historical\"");
// Add contract information for auto-detection
jsonBuilder.AppendFormat(",\"contract\":\"{0}\"", Instrument.MasterInstrument.Name);
```

### 2. Realtime Data Section (Multi-timeframe format)

**Location**: Around line 620-630 in realtime data sending

**Find this section** (in the multi-timeframe JSON building):
```csharp
jsonBuilder.AppendFormat("\"position_quantity\":{0}", positionQuantity);
jsonBuilder.Append("}");
```

**Change to**:
```csharp
jsonBuilder.AppendFormat("\"position_quantity\":{0},", positionQuantity);
// Add contract information for auto-detection
jsonBuilder.AppendFormat("\"contract\":\"{0}\"", Instrument.MasterInstrument.Name);
jsonBuilder.Append("}");
```

### 3. Realtime Data Section (Legacy format)

**Location**: Around line 650-665 in legacy format

**Find this line**:
```csharp
json = string.Format("{{\"time\":\"{0}\",\"open\":{1},\"high\":{2},\"low\":{3},\"close\":{4},\"volume\":{5},\"type\":\"realtime\",\"dailyPnL\":{6},\"dailyGoal\":{7},\"dailyMaxLoss\":{8},\"accountBalance\":{9},\"current_position\":\"{10}\",\"entry_price\":{11},\"position_quantity\":{12}}}",
```

**Change to**:
```csharp
json = string.Format("{{\"time\":\"{0}\",\"open\":{1},\"high\":{2},\"low\":{3},\"close\":{4},\"volume\":{5},\"type\":\"realtime\",\"dailyPnL\":{6},\"dailyGoal\":{7},\"dailyMaxLoss\":{8},\"accountBalance\":{9},\"current_position\":\"{10}\",\"entry_price\":{11},\"position_quantity\":{12},\"contract\":\"{13}\"}}",
```

**And update the parameters**:
```csharp
json = string.Format("{{\"time\":\"{0}\",\"open\":{1},\"high\":{2},\"low\":{3},\"close\":{4},\"volume\":{5},\"type\":\"realtime\",\"dailyPnL\":{6},\"dailyGoal\":{7},\"dailyMaxLoss\":{8},\"accountBalance\":{9},\"current_position\":\"{10}\",\"entry_price\":{11},\"position_quantity\":{12},\"contract\":\"{13}\"}}",
    barTime,
    barOpen,
    barHigh,
    barLow,
    barClose,
    barVolume,
    dailyPnL,
    dailyProfitGoal,
    dailyMaxLoss,
    accountBalance,
    currentPosition,
    entryPrice,
    positionQuantity,
    Instrument.MasterInstrument.Name);  // NEW: Contract name
```

## What This Does

**Before**: Server uses default contract (MNQ) from config.py
**After**: Server auto-detects contract from NinjaTrader

This allows you to:
- Switch between MNQ, ES, NQ, etc. in NinjaTrader
- Server automatically uses correct tick/point values
- No manual configuration needed when changing contracts

## Expected Instrument Names

The `Instrument.MasterInstrument.Name` property will return:
- "MNQ" for Micro E-mini NASDAQ
- "ES" for E-mini S&P 500
- "NQ" for E-mini NASDAQ
- "YM" for E-mini Dow
- "RTY" for E-mini Russell 2000
- "MES" for Micro E-mini S&P 500

## Testing

After making these changes:

1. Compile the strategy in NinjaTrader (F5)
2. Check for compilation errors
3. Enable the strategy on a chart
4. Check the RNN server logs - you should see:
   ```
   Contract: MNQ
   Tick Value: $0.5 | Point Value: $2.0
   ```

## Optional vs Required

**These changes are OPTIONAL** because:
- The server already defaults to MNQ in config.py
- It works correctly without these changes

**However, they are RECOMMENDED** if you:
- Trade multiple contracts (ES, MNQ, etc.)
- Want automatic detection
- Plan to switch contracts frequently

## Fallback Behavior

If you don't make these changes:
- Server uses `CONTRACT` value from `config.py` (currently "MNQ")
- This is perfectly fine if you only trade MNQ
- Just update `config.py` if you switch contracts

## Summary

| Change | Lines | Priority | Impact |
|--------|-------|----------|--------|
| Historical data | ~540 | Optional | Auto-detect on startup |
| Realtime MTF format | ~630 | Optional | Auto-detect per bar |
| Realtime legacy | ~665 | Optional | Auto-detect per bar |

All three changes are identical in concept: add contract name to JSON payload.
