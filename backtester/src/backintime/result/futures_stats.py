import typing as t
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, DivisionByZero

from backintime.broker.base import OrderSide
from backintime.broker.base import TradeInfo as Trade


@dataclass
class TradeProfit:
    trade_id: int
    order_id: int
    relative_profit: Decimal
    absolute_profit: Decimal


def _repr_profit(trade_profit: TradeProfit, percents_first=True) -> str:
    """
    Utility to represent `TradeProfit` object with control over
    the order of fields, for instance:
        Best deal (relative change): +14% (+5k absoulte) Trade#10 Order#15
        Best deal (absolute change): +20k (+10% relative) Trade#11 Order#20
    """
    if trade_profit is None:
        return repr(None)
    if percents_first:
        return (f"{trade_profit.relative_profit:+.2f}% "
                f"({trade_profit.absolute_profit:+.2f} absolute) "
                f"Trade#{trade_profit.trade_id} "
                f"Order#{trade_profit.order_id}")
    else:
        return (f"{trade_profit.absolute_profit:+.2f} "
                f"({trade_profit.relative_profit:+.2f}% relative) "
                f"Trade#{trade_profit.trade_id} "
                f"Order#{trade_profit.order_id}")


def _repr_percents(value: Decimal) -> str:
    """Represent decimal value in percents format."""
    return f"{value:+.2f}%" if not value.is_nan() else str(value)


@dataclass
class Stats:
    trades_profit: list
    profit_loss_ratio: Decimal = Decimal('NaN')
    profit_factor: Decimal = Decimal('NaN')
    win_rate: Decimal = Decimal('NaN')
    win_loss_ratio: Decimal = Decimal('NaN')
    wins_count: int = 0
    losses_count: int = 0
    expectancy: Decimal = Decimal(0)
    average_profit_all: Decimal = Decimal('NaN')
    average_profit_all_percents: Decimal = Decimal('NaN')
    average_profit: Decimal = Decimal('NaN')
    average_profit_percents: Decimal = Decimal('NaN')
    average_loss: Decimal = Decimal('NaN')
    average_loss_percents: Decimal = Decimal('NaN')
    best_deal_relative: t.Optional[TradeProfit] = None
    best_deal_absolute: t.Optional[TradeProfit] = None
    worst_deal_relative: t.Optional[TradeProfit] = None
    worst_deal_absolute: t.Optional[TradeProfit] = None

    def __repr__(self) -> str:
        best_deal_rel = _repr_profit(self.best_deal_relative)
        best_deal_abs = _repr_profit(self.best_deal_absolute, False)
        worst_deal_rel = _repr_profit(self.worst_deal_relative)
        worst_deal_abs = _repr_profit(self.worst_deal_absolute, False)

        win_rate = f"{self.win_rate:.2f}%" if not self.win_rate.is_nan() \
                    else str(self.win_rate) 
        avg_profit_percents = _repr_percents(self.average_profit_percents)
        avg_loss_percents = _repr_percents(self.average_loss_percents)
        avg_profit_all_percents = _repr_percents(self.average_profit_all_percents)

        return (f"Profit/Loss:\t{self.profit_loss_ratio:.2f}\n"
                f"Profit Factor:\t{self.profit_factor:.2f}\n"
                f"Win rate:\t{win_rate}\n"
                f"Win/Loss:\t{self.win_loss_ratio:.2f}\n"
                f"Wins count:\t{self.wins_count}\n"
                f"Losses count:\t{self.losses_count}\n"
                f"Expectancy:\t{self.expectancy:.2f}\n\n"
                f"Avg. Profit (all trades): {self.average_profit_all:+.2f}\n"
                f"Avg. Profit (all trades), %: {avg_profit_all_percents}\n"
                f"Avg. Profit (profit-making trades): {self.average_profit:+.2f}\n"
                f"Avg. Profit (profit-making trades), %: {avg_profit_percents}\n"
                f"Avg. Loss (loss-making trades): {self.average_loss:.2f}\n"
                f"Avg. Loss (loss-making trades), %: {avg_loss_percents}\n\n"
                f"Best deal (relative change): {best_deal_rel}\n"
                f"Best deal (absolute change): {best_deal_abs}\n"
                f"Worst deal (relative change): {worst_deal_rel}\n"
                f"Worst deal (absolute change): {worst_deal_abs}\n")


def get_trades_profit(trades):
    """Dummy algo for computing trades profit."""
    trades_profit = []
    trades = iter(trades)

    for fst in trades:
        try:
            snd = next(trades)
        except StopIteration:
            return trades_profit

        gain = snd.result_balance - fst.result_balance
        percents_gain = snd.result_balance/(fst.result_balance/100) - 100
        trade_profit=TradeProfit(trade_id=fst.trade_id, 
                                 order_id=fst.order.order_id,
                                 absolute_profit=gain,
                                 relative_profit=percents_gain)
        trades_profit.append(trade_profit)
    return trades_profit


def get_stats(trades: t.Sequence[Trade]) -> Stats:
    """
    Get stats such as Win Rate, Profit/Loss, Average Profit, etc.

    Return stats with default values for empty trades list 
    or for trades list without sells.
    """
    trades_profit = get_trades_profit(trades)
    stats = Stats(trades_profit=trades_profit)
    if not len(trades_profit):
        return stats

    wins_count = 0
    losses_count = 0
    total_gain = 0
    total_loss = 0

    acc_percents = 0
    acc_loss_percents = 0
    acc_profit_percents = 0

    best_absolute = worst_absolute = trades_profit[0]
    best_relative = worst_relative = trades_profit[0]

    for profit in trades_profit:
        if profit.absolute_profit > 0:
            wins_count += 1
            total_gain += profit.absolute_profit
            acc_profit_percents += profit.relative_profit

        elif profit.absolute_profit < 0:
            losses_count += 1 
            total_loss += abs(profit.absolute_profit)
            acc_loss_percents += abs(profit.relative_profit)

        acc_percents += profit.relative_profit
        best_relative = max(profit, best_relative,
                            key=lambda trade: trade.relative_profit)
        best_absolute = max(profit, best_absolute, 
                            key=lambda trade: trade.absolute_profit)
        worst_relative = min(profit, worst_relative,
                             key=lambda trade: trade.relative_profit)
        worst_absolute = min(profit, worst_absolute,
                             key=lambda trade: trade.absolute_profit)

    trades_count = len(trades)
    sell_trades_count = len(trades_profit)

    stats.wins_count = wins_count
    stats.losses_count = losses_count
    stats.best_deal_absolute = best_absolute
    stats.best_deal_relative = best_relative
    stats.worst_deal_absolute = worst_absolute
    stats.worst_deal_relative = worst_relative

    average_profit = total_gain/wins_count \
                        if wins_count else Decimal('NaN')

    stats.average_profit = average_profit
    stats.average_profit_percents = acc_profit_percents / wins_count \
                        if wins_count else Decimal('NaN')

    average_loss = total_loss/losses_count \
                        if losses_count else Decimal('NaN')

    stats.average_loss = -average_loss
    stats.average_loss_percents = - (acc_loss_percents / losses_count) \
                        if losses_count else Decimal('NaN')

    stats.average_profit_all = (total_gain - total_loss) / sell_trades_count \
                        if sell_trades_count else Decimal('NaN')

    stats.average_profit_all_percents = acc_percents / sell_trades_count \
                        if sell_trades_count else Decimal('NaN')

    stats.profit_loss_ratio = average_profit/average_loss \
                        if average_loss else Decimal('NaN')

    stats.profit_factor = total_gain/total_loss \
                        if total_loss else Decimal('NaN')

    stats.win_loss_ratio = Decimal(wins_count/losses_count) \
                        if losses_count else Decimal('NaN')

    stats.win_rate = Decimal(wins_count/(sell_trades_count/100)) \
                        if sell_trades_count else Decimal('NaN')
    
    win_pct = Decimal(wins_count/sell_trades_count) \
                        if sell_trades_count else Decimal('NaN')

    loss_pct = Decimal(losses_count/sell_trades_count) \
                        if sell_trades_count else Decimal('NaN')

    stats.expectancy = win_pct * average_profit - loss_pct * average_loss

    return stats

