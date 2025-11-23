"""
✨ Tool for testing trading strategies on historical data ✨    
Such testing does not guarantee the same results in real trading, 
but it gives a rough estimate of a strategy's success.
The smaller timeframe is used, the more accurate the results will be.
Short overview of the core concepts is given below.


Backtesting

Backtesting is done in `run_backtest` function, which basically 
initializes required objects, prefetches market data if needed 
and runs backtesting on historical data.


TradingStrategy

Base class for trading strategies. 
Strategy must provide algorithm implementation in `tick` method, 
which runs each time a new candle closes.


Broker

Broker provides orders management in a simulated
market environment. The broker executes/activates orders
whose conditions fits the market. Supports Market, Limit,
Take Profit, Take Profit Limit, Stop Loss, Stop Loss Limit orders.
Order execution policy of builtin broker:

    - Market orders: 
        All market orders will be executed 
        when a new candle closes. 
        The price of execution is the candle's OPEN price.

    - Limit orders: 
        Limit order will be executed at the limit price or better:
        lower or equal price for BUY orders and higher or equal price 
        for SELL orders.
        First, the `order_price` of each order will be compared to
        the OPEN price of a new candle: 
            BUY orders will be executed if `order_price` >= OPEN. 
            SELL orders will be executed if `order_price` <= OPEN.

        Then, remaining BUYs will be compared to LOW,
        and remaining SELLs - to HIGH.
        Fill price is the first price that matched limit price.

    - Take Profit/Stop Loss orders: 
        TP/SL orders will be activated if the `trigger_price` is
        within the price bounds of a candle.

        This check is performed in two steps:
            1) For each order: activate if trigger_price == OPEN
            2) For each order: activate if LOW <= trigger_price <= HIGH 

        When a TP/SL order is triggered, it will be treated
        as a market or limit order, depending on whether 
        `order_price` is set for the order.

    Limit, Take Profit and Stop Loss orders are reviewed 
    in the order of their submission (oldest first).


Analyser

Indicators calculation.
Supported indicators:
    - MA - Moving Average
    - EMA - Exponential Moving Average
    - MACD - Moving Average Convergence Divergence
    - BBANDS - Bollinger Bands
    - RSI - Relative Strength Index
    - ATR - Average True Range
    - ADX - Average Directional Movement Index
    - DMI - Directional Movement Indicator
    - PIVOT - Pivot Points (Traditional, Fibonacci or Classic)


Candles

Provides the last candle representation for various timeframes.
It is useful for checking properties of a candle 
on one timeframe (H1, for example), while having data
on another (for instance, M1).


DataProvider

Provides candles in historical order.
`DataProvider` is an iterable object that 
can be created for specific date range (since, until);
Yields OHLCV candle during iteration.


BacktestingResult

Provides export to CSV and stats such as Win Rate, Profit/Loss,
Average Profit, Best/Worst deal etc.
Supports estimation in FIFO (First-In-First-Out), 
LIFO (Last-In-First-Out) or AVCO (Average Cost) algorithms.
The algorithm name specifies the order in which BUYs
must be considered to estimate profit or loss.

All these algorithms produce the same result if SELL
order always follows only one BUY with the same amount.


Prefetching

Indicators require a certain amount of data to get a correct result. 
For example, to calculate the SMA (simple moving average) with 
a period of 9, 9 values are required. 
So, the strategy will get the wrong result of the SMA indicator, 
until all 9 candles are accumulated.

In order for the strategy to get the correct values 
right from the start, prefetching of market data is used.

You can configure this behavior by choosing from the following 
options and passing it as `prefetch_option` argument to 
`run_backtest` function:

    - PREFETCH_UNTIL (default): 
        Prefetch the required amount of data until `since`* date; 
        the amount of data required is calculated automatically. 
        In this case, the strategy backtesting will be started 
        from the `since` date. 
        This option is convenient when market data is requested 
        from the exchange API, because situations when the exchange
        does not have enough historical data are quite rare.
        
    - PREFETCH_SINCE:
        Prefetch the required amount of data from `since`* date; 
        the amount of data required is calculated automatically. 
        In this case, the strategy backtesting will be launched 
        starting from the dynamically calculated date. 
        This option may be useful when working with a CSV file 
        when you are not sure if it contains enough data before 
        the `since` date.
        
    - PREFETCH_NONE - Do not prefetch data.  

*`since` date is the value of the argument `since` passed to 
the `run_backtest` function. 
"""
import logging

from .trading_strategy import FuturesStrategy
from .utils import run_backtest

# Configure logging
logging.basicConfig(level='INFO')
