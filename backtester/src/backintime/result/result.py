import os
import re
import typing as t
from datetime import datetime, timezone
from decimal import Decimal

from backintime.broker.base import OrderInfo, TradeInfo
from backintime.data.data_provider import DataProvider

from .csv import export_orders, export_stats, export_trades
from .futures_stats import Stats, TradeProfit, get_stats
import pandas as pd

class BacktestingStats:
    def __init__(self, 
                 strategy_title: str,
                 data_provider: DataProvider,
                 date: datetime,
                 stats: Stats):
        self._strategy_title = strategy_title
        self._data_title = data_provider.title
        self._data_timeframe = data_provider.timeframe
        self._data_since = data_provider.since
        self._data_until = data_provider.until
        self._symbol = data_provider.symbol
        self._date = date
        self._stats = stats

    @property
    def strategy_title(self) -> str:
        return self._strategy_title

    @property
    def date(self) -> datetime:
        return self._date

    @property
    def algorithm(self) -> str:
        return self._stats.algorithm

    @property
    def average_profit(self) -> Decimal:
        return self._stats.average_profit

    @property
    def profit_loss_ratio(self) -> Decimal:
        return self._stats.profit_loss_ratio

    @property
    def profit_factor(self) -> Decimal:
        return self._stats.profit_factor

    @property
    def win_rate(self) -> Decimal:
        return self._stats.win_rate

    @property
    def win_loss_ratio(self) -> Decimal:
        return self._stats.win_loss_ratio

    @property
    def wins_count(self) -> int:
        return self._stats.wins_count

    @property
    def losses_count(self) -> int:
        return self._stats.losses_count

    @property
    def average_profit_all(self) -> Decimal:
        """Average profit for all trades."""
        return self._stats.average_profit_all

    @property
    def average_profit_all_percents(self) -> Decimal:
        """Average profit for all trades in percents."""
        return self._stats.average_profit_all_percents

    @property
    def average_profit(self) -> Decimal:
        """Average profit for profit-making trades."""
        return self._stats.average_profit

    @property
    def average_profit_percents(self) -> Decimal:
        """Average profit for profit-making trades in percents."""
        return self._stats.average_profit_percents

    @property
    def average_loss(self) -> Decimal:
        """Average loss for loss-making trades."""
        return self._stats.average_loss

    @property
    def average_loss_percents(self) -> Decimal:
        """Average loss for loss-making trades in percents."""
        return self._stats.average_loss_percents

    @property
    def best_deal_relative(self) -> t.Optional[TradeProfit]:
        """Best deal by relative gain."""
        return self._stats.best_deal_relative

    @property
    def best_deal_absolute(self) -> t.Optional[TradeProfit]:
        """Best deal by absolute gain."""
        return self._stats.best_deal_absolute

    @property
    def worst_deal_relative(self) -> t.Optional[TradeProfit]:
        """Worst deal by relative loss."""
        return self._stats.worst_deal_relative

    @property
    def worst_deal_absolute(self) -> t.Optional[TradeProfit]:
        """Worst deal by absolute loss."""
        return self._stats.worst_deal_absolute

    def __repr__(self) -> str:
        date = datetime.strftime(self._date, "%Y-%m-%d %H:%M:%S")
        header_message = f"Backtesting Stats {date}"
        header = f"\n{'-' * 16}* {header_message} *{'-' * 17}\n\n"
        footer = f"\n{'-' * 74}\n"
        data_block = (f"{self._data_title} on {str(self._data_timeframe)}\n"
                      f"since: {self._data_since}\n"
                      f"until: {self._data_until}\n"
                      f"Trading Pair: {self._symbol}\n\n")

        return (f"{header}{data_block}"
                f"Strategy title: {self._strategy_title}\n"
                f"{repr(self._stats)}{footer}")


class BacktestingResult:
    """
    Represents backtesting result.

    Stats such as Win Rate, Profit/Loss, Average Profit
    can be obtained by calling `get_stats` method.

    Orders, trades and stats can be exported to CSV file
    using `export_orders`, `export_trades`, `exports_stats`
    repectively, or via `export` - for exporting all 
    with default file names.
    """
    def __init__(self, 
                 strategy_title: str,
                 strategy_params: str,
                 strategy_indicators: str,
                 data_provider: DataProvider,
                 start_balance: Decimal,
                 result_balance: Decimal,
                 result_equity: Decimal,
                 min_equity: Decimal,
                 max_equity: Decimal,
                 trades: t.Sequence[TradeInfo],
                 orders: t.Sequence[OrderInfo]):
        self._data_provider = data_provider
        self._data_title = data_provider.title
        self._data_timeframe = data_provider.timeframe
        self._data_since = data_provider.since
        self._data_until = data_provider.until
        self._symbol = data_provider.symbol
        self._strategy_title = strategy_title
        self._strategy_params = strategy_params
        self._strategy_indicators = strategy_indicators
        self._start_balance = start_balance
        self._result_balance = result_balance
        self._result_equity = result_equity
        self._min_equity = min_equity
        self._max_equity = max_equity
        diff = result_equity - start_balance
        self._total_gain = diff
        self._total_gain_percents = diff/(start_balance/100)
        self._trades = trades
        self._orders = orders
        self._trades_count = len(trades)
        self._orders_count = len(orders)
        self._date = datetime.now()

    @property
    def strategy_title(self) -> str:
        return self._strategy_title

    @property
    def strategy_params(self) -> str:
        return self._strategy_params

    @property
    def strategy_indicators(self) -> str:
        return self._strategy_indicators

    @property
    def date(self) -> datetime:
        return self._date

    @property
    def start_balance(self) -> Decimal:
        return self._start_balance

    @property
    def result_balance(self) -> Decimal:
        return self._result_balance

    @property
    def result_equity(self) -> Decimal:
        return self._result_equity

    @property
    def min_equity(self) -> Decimal:
        return self._min_equity

    @property
    def max_equity(self) -> Decimal:
        return self._max_equity

    @property
    def total_gain(self) -> Decimal:
        return self._total_gain

    @property
    def total_gain_percents(self):
        return self._total_gain_percents

    @property
    def trades_count(self) -> int:
        return self._trades_count

    @property
    def orders_count(self) -> int:
        return self._orders_count

    def get_stats(self) -> BacktestingStats:
        return BacktestingStats(self._strategy_title, 
                                self._data_provider, 
                                self._date, 
                                get_stats(self._trades))

    def export(self) -> None:
        """Export stats, trades and orders to CSV files."""
        self.export_stats()
        self.export_trades()
        self.export_orders()

    def export_stats(self,
                     path='.',
                     filename: t.Optional[str] = None,
                     delimiter=';',
                     optimization=False,
                     lock = None) -> None:
        """Export stats to CSV file."""
        filename = filename or self._get_default_csv_filename('stats')
        filepath = os.path.join(path, filename)
        stats = get_stats(self._trades)

        # print(f'result filepath: {filepath}')
        export_stats(filepath, delimiter, 
                     self.strategy_title, 
                     self.strategy_params, self.strategy_indicators,
                     self.date, self._data_title, self._data_timeframe, 
                     self._symbol, self._data_since, self._data_until, 
                     self.start_balance, self.result_balance,
                     self.min_equity, self.max_equity, self.result_equity,
                     self.total_gain, self.total_gain_percents, stats, 
                     optimization, lock)

    def export_trades(self, 
                      path='.',
                      filename: t.Optional[str] = None,
                      delimiter=';') -> None:
        """Export trades to CSV file."""
        filename = filename or self._get_default_csv_filename('trades')
        filepath = os.path.join(path, filename)
        export_trades(filepath, delimiter, self._trades)

    def export_orders(self, 
                      path='.',
                      filename: t.Optional[str] = None,
                      delimiter=';') -> None:
        """Export orders to CSV file."""
        filename = filename or self._get_default_csv_filename('orders')
        filepath = os.path.join(path, filename)
        export_orders(filepath, delimiter, self._orders)

    def _get_default_csv_filename(self, entity: str) -> str:
        """
        Get filename for exporting something that is called `entity`
        including .csv extension in the end.
        """
        strategy_title = self._strategy_title.lower()
        strategy_title = '_'.join(strategy_title.split())
        strategy_title = re.sub(r'[\\, /]', '_', strategy_title)
        date_postfix = datetime.strftime(self._date, "%Y%m%d%H%M%S")
        return f"{strategy_title}_{entity}_{date_postfix}.csv"

    def __repr__(self) -> str:
        date = datetime.strftime(self._date, "%Y-%m-%d %H:%M:%S")
        header_message = f"Backtesting Result {date}"
        header = f"\n{'-' * 16}* {header_message} *{'-' * 16}\n\n"
        footer = f"\n{'-' * 74}\n"
        data_block = (f"{self._data_title} on {str(self._data_timeframe)}\n"
                      f"since: {self._data_since}\n"
                      f"until: {self._data_until}\n"
                      f"Trading Pair: {self._symbol}\n\n")

        content = (f"Strategy title: {self.strategy_title}\n"
                   f"Strategy params: {self.strategy_params}\n"
                   f"Start balance:\t\t{self.start_balance:.2f}\n"
                   f"Result balance:\t\t{self.result_balance:.2f}\n"
                   f"Min equity:\t\t{self.min_equity:.2f}\n"
                   f"Max equity:\t\t{self.max_equity:.2f}\n"
                   f"Result equity:\t\t{self.result_equity:.2f}\n"
                   f"Total gain:\t\t{self.total_gain:.2f}\n"
                   f"Total gain percents:\t{self.total_gain_percents:.2f}%\n"
                   f"Trades count:\t{self.trades_count}\n"
                   f"Orders count:\t{self.orders_count}\n")

        return f"{header}{data_block}{content}{footer}"

def rank_by_stat(filepath: str, sort_column: str = 'Expectancy', ranking: int = 10, additional_info: dict = {}):
    """
    Sort the stats of an optimization run. 
    
    Return sorted stats and print out the ranking based on sort_column
    """
    
    df = pd.read_csv(filepath)
    
    # Sort by descending order (largest to smallest)
    sorted_df = df.sort_values(by=sort_column, ascending=False)
    
    # Split the filename at the last _ and insert the ranked column name
    prefix, suffix = filepath.rsplit('_', 1)
    ranked_filepath = f"{prefix}_ranked_{sort_column.lower()}_{suffix}"
    sorted_df.to_csv(ranked_filepath, index=False)
    
    date = datetime.strftime(additional_info['date'], "%Y-%m-%d %H:%M:%S")
    header_message = f"Backtesting Stats Ranking {date}"
    header = f"\n{'-' * 16}* {header_message} *{'-' * 17}\n\n"
    footer = f"\n{'-' * 74}\n"
    data_block = (f"{additional_info['data_title']} on {str(additional_info['data_timeframe'])}\n"
                    f"since: {additional_info['data_since']}\n"
                    f"until: {additional_info['data_until']}\n"
                    f"Trading Pair: {additional_info['symbol']}\n"
                    f"Ranked Stat: {sort_column}\n\n")

    display_rank_df = sorted_df[['Strategy Title','Strategy Params','Result Balance',sort_column]]
    print (f"{header}{data_block}"
            f"Strategy title: {additional_info['strategy_title']}\n"
            f"{display_rank_df[:ranking]}{footer}")