# Contract Support

The RNN trading system supports multiple futures contracts through the `contract_specs.py` module.

## Supported Contracts

### Standard E-mini Contracts
- **ES** - E-mini S&P 500 ($50/point, 0.25 tick)
- **NQ** - E-mini NASDAQ-100 ($20/point, 0.25 tick)
- **YM** - E-mini Dow ($5/point, 1.0 tick)
- **RTY** - E-mini Russell 2000 ($50/point, 0.10 tick)

### Micro E-mini Contracts
- **MES** - Micro E-mini S&P 500 ($5/point, 0.25 tick)
- **MNQ** - Micro E-mini NASDAQ-100 ($2/point, 0.25 tick)

## Usage

### Get Contract Specifications

```python
from contract_specs import get_contract

# Get ES specs
es = get_contract('ES')
print(f"ES tick value: ${es.tick_value}")
print(f"ES point value: ${es.point_value}")

# Get NQ specs
nq = get_contract('NQ')
print(f"NQ tick value: ${nq.tick_value}")
print(f"NQ point value: ${nq.point_value}")
```

### Calculate P&L

```python
from contract_specs import calculate_pnl

# ES trade
pnl_es = calculate_pnl(
    entry_price=4800.0,
    exit_price=4810.0,
    direction='long',
    contracts=1,
    symbol='ES'
)
# Returns: $500.00 (10 points * $50/point)

# NQ trade
pnl_nq = calculate_pnl(
    entry_price=17000.0,
    exit_price=17050.0,
    direction='long',
    contracts=1,
    symbol='NQ'
)
# Returns: $1000.00 (50 points * $20/point)
```

### Calculate Margin

```python
from contract_specs import calculate_margin_requirement

# Day trading margin
day_margin = calculate_margin_requirement(2, 'ES', overnight=False)
# Returns: $25,000.00

# Overnight margin
overnight_margin = calculate_margin_requirement(2, 'NQ', overnight=True)
# Returns: $35,200.00
```

### List All Contracts

```python
from contract_specs import list_contracts

list_contracts()
# Displays all available contracts with specs
```

## Using with Backtester

### RNN Backtester

Update the backtester to use contract specs:

```python
from backtester import Backtester
from contract_specs import get_contract

# Get NQ specs
nq = get_contract('NQ')

# Create backtester with NQ values
backtester = Backtester(
    initial_capital=25000.0,
    commission_per_contract=2.50,
    slippage_ticks=1,
    tick_value=nq.tick_value,      # $5.00 for NQ
    daily_goal=500.0,
    daily_max_loss=250.0
)
```

### backintime Framework

Use contract specs in the adapter:

```python
from backintime_rnn_adapter import RNNFuturesStrategy
from contract_specs import get_contract

class NQRNNStrategy(RNNFuturesStrategy):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

        # Override with NQ specs
        nq = get_contract('NQ')
        self.tick_size = nq.tick_size
        self.point_value = nq.point_value
```

## Sample Data Generation

Generate sample data for different contracts:

```python
from sample_data.generate_sample_data import generate_realistic_es_data
from contract_specs import get_contract

# Generate NQ data
nq = get_contract('NQ')

df = generate_realistic_es_data(
    n_days=10,
    start_date="2024-01-02",
    start_price=17000.0,  # NQ typical price
    output_dir="nq_sample_data"
)

# Adjust to NQ tick size (already 0.25, same as ES)
```

## Current Default

By default, the system is configured for **ES (E-mini S&P 500)**:
- Tick size: 0.25
- Point value: $50
- Tick value: $12.50

To use **NQ** or other contracts, update the relevant configuration files or pass contract specs to the backtester/strategy constructors.

## Migration Plan

To make contract selection easier in the future:

1. **Add contract parameter to Backtester**:
   ```python
   backtester = Backtester(
       initial_capital=25000.0,
       contract='NQ'  # Auto-loads NQ specs
   )
   ```

2. **Add contract parameter to sample data generator**:
   ```python
   generate_realistic_es_data(
       n_days=10,
       contract='NQ'  # Generates NQ-style data
   )
   ```

3. **Update strategy adapter**:
   ```python
   run_rnn_backtest(
       model=model,
       contract='NQ',  # Uses NQ specs
       ...
   )
   ```

## Margin Notes

**Important**: Margin requirements change periodically. The values in `contract_specs.py` are approximate. Always check with your broker for current requirements:

- CME Group: https://www.cmegroup.com/trading/equity-index/
- Your broker's margin page

## Adding New Contracts

To add a new contract:

1. Add to `CONTRACTS` dict in `contract_specs.py`
2. Specify all required fields
3. Test with sample data
4. Update this README

Example:
```python
'CL': ContractSpec(
    symbol='CL',
    name='Crude Oil',
    tick_size=0.01,
    point_value=1000.0,
    tick_value=10.0,
    # ... other specs
)
```
