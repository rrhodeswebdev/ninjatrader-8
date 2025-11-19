# QUICKSTART: Trading MNQ with RNN Server

Your RNN server is now configured for **MNQ (Micro E-mini NASDAQ-100)** trading.

## What Changed

✅ **Default contract set to MNQ** in `config.py`
- Tick value: $0.50 (was $12.50)
- Point value: $2 (was $50)

✅ **Risk management now uses correct MNQ values**
- Position sizing calculations fixed
- Stop loss/take profit distances correct

## Immediate Action Required

The system currently defaults to MNQ values. However, for **complete auto-detection**, you have two options:

### Option 1: Use Config Default (Works Now)

No changes needed. The system uses MNQ by default from `config.py`.

### Option 2: Send Contract from NinjaTrader (Recommended)

Update `AITrader.cs` to send the instrument name:

**In the historical data section** (around line 520):
```csharp
jsonBuilder.Append($",\"contract\":\"{Instrument.MasterInstrument.Name}\"");
```

**In the realtime data section** (around line 650):
```csharp
jsonBuilder.Append($",\"contract\":\"{Instrument.MasterInstrument.Name}\"");
```

This will automatically send "MNQ", "ES", or whatever you're trading.

## Verify Configuration

Check your current setup:

```bash
cd rnn-server
python3 -c "from config import CONTRACT; print(f'Default contract: {CONTRACT}')"
# Should output: Default contract: MNQ

python3 -c "from contract_specs import get_contract; spec = get_contract('MNQ'); print(f'MNQ tick value: ${spec.tick_value}')"
# Should output: MNQ tick value: $0.50
```

## What This Fixes

**Before** (ES values):
- Risk per contract = Stop distance × 4 ticks × $12.50 = WAY TOO HIGH
- Position size = Risk / Risk per contract = TOO SMALL

**After** (MNQ values):
- Risk per contract = Stop distance × 4 ticks × $0.50 = CORRECT
- Position size = Risk / Risk per contract = CORRECT

### Example

Account: $25,000
Risk: 1% = $250
Stop: 10 points (40 ticks)

**Before** (wrong ES values):
- Risk per contract = 10 × 4 × $12.50 = $500
- Position size = $250 / $500 = 0 contracts ❌

**After** (correct MNQ values):
- Risk per contract = 10 × 4 × $0.50 = $20
- Position size = $250 / $20 = 12 contracts ✅

## Next Steps

1. ✅ System configured for MNQ
2. ⚠️ **Optional**: Update NinjaTrader C# to send contract name
3. ✅ Start trading with correct position sizes!

## Switching Contracts

To trade ES instead:

```python
# Edit config.py:
CONTRACT = "ES"  # Change from "MNQ"
```

Or send `contract` field from NinjaTrader (Option 2 above).

## Support for Multiple Contracts

Currently supported:
- **ES** - E-mini S&P 500 ($50/point, $12.50/tick)
- **NQ** - E-mini NASDAQ-100 ($20/point, $5/tick)
- **MNQ** - Micro NASDAQ ($$2/point, $0.50/tick) ← **Your current setup**
- **MES** - Micro S&P 500 ($5/point, $1.25/tick)
- **YM** - E-mini Dow ($5/point, $5/tick)
- **RTY** - E-mini Russell 2000 ($50/point, $5/tick)

See `contract_specs.py` for full details.

## Questions?

- Config file: `rnn-server/config.py`
- Contract specs: `rnn-server/contract_specs.py`
- Documentation: `rnn-server/CONTRACT_SUPPORT.md`
