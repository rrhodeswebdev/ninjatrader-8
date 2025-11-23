"""CSV export for backintime entities: stats, orders and trades."""
import os
import csv
import typing as t
from datetime import datetime
from decimal import Decimal

from backintime.broker.base import OrderInfo, TradeInfo
from backintime.timeframes import Timeframes

from .futures_stats import Stats


def decimal_to_str(value: Decimal) -> str:
    """Covert decimal value to str with 4fp precision."""
    return str(value.quantize(Decimal('0.0001'))) if not value.is_nan() else ''


def datetime_to_str(value: datetime) -> str:
    """
    Represent datetime in ISO-8601 format 
    with date and time separated by space.
    """
    return value.isoformat(sep=' ')


def export_stats(filename: str,
                 delimiter: str,
                 strategy_title: str,
                 strategy_params: str,
                 strategy_indicators: str,
                 date: datetime,
                 data_title: str,
                 data_timeframe: Timeframes,
                 data_symbol: str,
                 data_since: datetime,
                 data_until: datetime,
                 start_balance: Decimal,
                 result_balance: Decimal,
                 min_equity: Decimal,
                 max_equity: Decimal,
                 result_equity: Decimal,
                 total_gain: Decimal,
                 total_gain_percents: Decimal,
                 stats: Stats,
                 optimization: bool,
                 lock) -> None:
    
# During mulitprocessing, need to lock the thread in order to safely read timestamp's value as well as append to the 
# same csv file
    """Export stats to CSV file."""
    if optimization:
        with lock:
            directory = os.path.dirname(filename)
            prefix, suffix = filename.rsplit('_', 1)
            print(f'csv filename: {filename}')
            print(f'csv prefix: {prefix}')
            with open(os.path.join(directory, 'timestamp.txt'),'r') as tsfile:
                filename = prefix+tsfile.readline()+'.csv'
            file_exists = False
            if os.path.exists(filename):
                file_exists = True
    else:
        file_exists = False
        if os.path.exists(filename):
            file_exists = True
    if optimization:
        lock.acquire()
    print(f'csv filename post: {filename}')
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter,
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # Write headers if not in optimization mode or if in optimization mode but header was
        # Never written(file did not exist)
        if not file_exists:
            writer.writerow([
                'Strategy Title',
                'Strategy Params',
                'Strategy Indicators',
                'Date',
                'Data Provider',
                'Timeframe',
                'Symbol',
                'Since',
                'Until',
                'Start Balance',
                'Result Balance',
                'Min Equity',
                'Max Equity',
                'Result Equity',
                'Total gain',
                'Total gain (%)',
                'Profit/Loss ratio',
                'Profit Factor',
                'Win/Loss ratio',
                'Win rate',
                'Wins count',
                'Losses count',
                'Expectancy',
                'Average Profit (all trades)',
                'Average Profit (all trades), %',
                'Average Profit (profit-making trades)',
                'Average Profit (profit-making trades), %',
                'Average Loss (loss-making trades)',
                'Average Loss (loss-making trades), %',
                'Best deal (relative)',
                'Best deal (absolute)',
                'Worst deal (relative)',
                'Worst deal (absolute)',
            ])
        # Write content
        writer.writerow([
                # Args
                strategy_title,
                strategy_params,
                strategy_indicators,
                datetime_to_str(date),
                data_title,
                data_timeframe,
                data_symbol,
                datetime_to_str(data_since),
                datetime_to_str(data_until),
                start_balance,
                result_balance,
                min_equity,
                max_equity,
                result_equity,
                total_gain,
                total_gain_percents,
                # Stats item
                decimal_to_str(stats.profit_loss_ratio),
                decimal_to_str(stats.profit_factor),
                decimal_to_str(stats.win_loss_ratio),
                decimal_to_str(stats.win_rate),
                stats.wins_count,
                stats.losses_count,
                stats.expectancy,
                decimal_to_str(stats.average_profit_all),
                decimal_to_str(stats.average_profit_all_percents),
                decimal_to_str(stats.average_profit),
                decimal_to_str(stats.average_profit_percents),
                decimal_to_str(stats.average_loss),
                decimal_to_str(stats.average_loss_percents),
                decimal_to_str(stats.best_deal_relative.relative_profit),
                decimal_to_str(stats.best_deal_absolute.absolute_profit),
                decimal_to_str(stats.worst_deal_relative.relative_profit),
                decimal_to_str(stats.worst_deal_absolute.absolute_profit)
        ])
    if optimization:
        lock.release()

def export_orders(filename: str,
                  delimiter: str,
                  orders: t.Sequence[OrderInfo]) -> None:
    """
    Export orders to CSV file.
    Won't take effect if `orders` is empty.
    """
    if not len(orders): 
        return

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter,
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # Write headers
        writer.writerow([
            "Order ID",
            "Order Type",
            "Side",
            "Short",
            "Amount",
            "Date Created",
            "Order Price",
            "Trigger Price",
            "Status",
            "Date Updated",
            "Fill Price",
            "Trading Fee"
        ])
        # Write content
        for order in orders:
            writer.writerow([
                order.order_id,
                order.order_type,
                order.order_side,
                order.is_short,
                order.amount,
                datetime_to_str(order.date_created),
                order.order_price,
                decimal_to_str(order.trigger_price),
                order.status,
                datetime_to_str(order.date_updated),
                order.fill_price,
                order.trading_fee,
            ])


def export_trades(filename: str,
                  delimiter: str,
                  trades: t.Sequence[TradeInfo]) -> None:
    """
    Export trades to CSV file.
    Won't take effect if `trades` is empty.
    """
    if not len(trades): 
        return

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter,
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # Write headers
        writer.writerow([
            "Order ID",
            "Order Type",
            "Side",
            "Short",
            "Amount",
            "Date Created",
            "Order Price",
            "Trigger Price",
            "Status",
            "Date Updated",
            "Fill Price",
            "Trading Fee",
            "Result Balance"
        ])
        # Write content
        for trade in trades:
            writer.writerow([
                trade.order.order_id,
                trade.order.order_type,
                trade.order.order_side,
                trade.order.is_short,
                trade.order.amount,
                datetime_to_str(trade.order.date_created),
                trade.order.order_price,
                decimal_to_str(trade.order.trigger_price),
                trade.order.status,
                datetime_to_str(trade.order.date_updated),
                trade.order.fill_price,
                trade.order.trading_fee,
                trade.result_balance
            ])
