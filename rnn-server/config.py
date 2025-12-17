# Trading System Configuration

# Contract Settings
# Supported: ES, NQ, YM, RTY, MES, MNQ
CONTRACT = "MNQ"  # Default contract for live trading

# Server Settings
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# Model Settings
MODEL_SEQUENCE_LENGTH = 15
MIN_CONFIDENCE_THRESHOLD = 0.55

# Risk Management
DAILY_GOAL = 100.0
DAILY_MAX_LOSS = 100.0
MAX_TRADES_PER_DAY = 10

# Backtesting
BACKTEST_INITIAL_CAPITAL = 25000.0
# Commission per contract (round-trip). Actual Tradovate: $0.35/side = $0.70 round-trip
BACKTEST_COMMISSION_PER_CONTRACT = 0.70
BACKTEST_SLIPPAGE_TICKS = 1
