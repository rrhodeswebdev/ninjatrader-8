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
DAILY_GOAL = 500.0
DAILY_MAX_LOSS = 250.0
MAX_TRADES_PER_DAY = 10

# Backtesting
BACKTEST_INITIAL_CAPITAL = 25000.0
BACKTEST_COMMISSION_PER_CONTRACT = 2.50
BACKTEST_SLIPPAGE_TICKS = 1
