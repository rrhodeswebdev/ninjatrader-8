import numpy as np
import pandas as pd
from typing import List, Tuple

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio for trading returns

    Args:
        returns: Array of returns (not cumulative)
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Number of trading periods per year (252 for daily, 252*390 for 1-min)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)

    if np.std(returns) == 0:
        return 0.0

    sharpe = np.mean(excess_returns) / np.std(returns)

    # Annualize
    return sharpe * np.sqrt(periods_per_year)


def calculate_profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor (gross profit / gross loss)

    Args:
        returns: Array of trade returns

    Returns:
        Profit factor (>1 is good, <1 is bad)
    """
    if len(returns) == 0:
        return 0.0

    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown

    Args:
        cumulative_returns: Array of cumulative returns

    Returns:
        Maximum drawdown as a percentage (negative number)
    """
    if len(cumulative_returns) == 0:
        return 0.0

    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / (running_max + 1e-8)

    return np.min(drawdown)


def calculate_win_rate(returns: np.ndarray) -> float:
    """
    Calculate win rate (% of profitable trades)

    Args:
        returns: Array of trade returns

    Returns:
        Win rate as decimal (0.0 to 1.0)
    """
    if len(returns) == 0:
        return 0.0

    wins = np.sum(returns > 0)
    return wins / len(returns)


def calculate_expectancy(returns: np.ndarray) -> float:
    """
    Calculate expectancy (average win * win_rate - average loss * loss_rate)

    Args:
        returns: Array of trade returns

    Returns:
        Expected value per trade
    """
    if len(returns) == 0:
        return 0.0

    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]

    if len(winning_trades) == 0:
        avg_win = 0
        win_rate = 0
    else:
        avg_win = np.mean(winning_trades)
        win_rate = len(winning_trades) / len(returns)

    if len(losing_trades) == 0:
        avg_loss = 0
        loss_rate = 0
    else:
        avg_loss = abs(np.mean(losing_trades))
        loss_rate = len(losing_trades) / len(returns)

    return (avg_win * win_rate) - (avg_loss * loss_rate)


def calculate_sortino_ratio(returns: np.ndarray, target_return: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (like Sharpe but only penalizes downside volatility)

    Args:
        returns: Array of returns
        target_return: Target/minimum acceptable return
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0

    downside_std = np.std(downside_returns)

    if downside_std == 0:
        return 0.0

    sortino = np.mean(excess_returns) / downside_std

    # Annualize
    return sortino * np.sqrt(periods_per_year)


def evaluate_trading_performance(predictions: List[int], actual_labels: List[int],
                                 price_changes: np.ndarray, daily_pnl_data: dict = None) -> dict:
    """
    Comprehensive trading performance evaluation with daily P&L limits

    Args:
        predictions: Model predictions (0=short, 1=hold, 2=long)
        actual_labels: Actual labels
        price_changes: Actual price changes for each prediction
        daily_pnl_data: Optional dict with 'dailyGoal' and 'dailyMaxLoss' for realistic simulation

    Returns:
        Dictionary of trading metrics
    """
    predictions = np.array(predictions)
    actual_labels = np.array(actual_labels)

    # Simulate trade returns based on predictions with daily P&L limits
    simulated_returns = []
    daily_pnl = 0.0
    trades_stopped_by_goal = 0
    trades_stopped_by_loss = 0

    # Extract daily limits if provided
    if daily_pnl_data:
        daily_goal = daily_pnl_data.get('dailyGoal', float('inf'))
        daily_max_loss = daily_pnl_data.get('dailyMaxLoss', float('inf'))
    else:
        daily_goal = float('inf')
        daily_max_loss = float('inf')

    for pred, price_change in zip(predictions, price_changes):
        # Check if we've hit daily limits (stop trading for the "day")
        # In real backtesting, you'd reset this on actual day boundaries
        if daily_pnl >= daily_goal:
            trades_stopped_by_goal += 1
            continue  # Skip this trade - daily goal reached

        if daily_pnl <= -daily_max_loss:
            trades_stopped_by_loss += 1
            continue  # Skip this trade - daily max loss hit

        # Execute trade
        trade_return = 0.0
        if pred == 2:  # Long
            trade_return = price_change
        elif pred == 0:  # Short
            trade_return = -price_change
        # Hold (1) generates no return

        if trade_return != 0:
            simulated_returns.append(trade_return)
            daily_pnl += trade_return

            # Check if this trade pushed us over the limits
            if daily_pnl >= daily_goal:
                trades_stopped_by_goal += 1
            elif daily_pnl <= -daily_max_loss:
                trades_stopped_by_loss += 1

    simulated_returns = np.array(simulated_returns)

    # Filter out holds for return-based metrics
    trading_returns = simulated_returns[simulated_returns != 0]

    if len(trading_returns) == 0:
        return {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'expectancy': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0,
            'num_trades': 0,
            'accuracy': 0.0,
            'trades_stopped_by_goal': 0,
            'trades_stopped_by_loss': 0
        }

    cumulative_returns = np.cumsum(trading_returns)

    # Calculate all metrics
    metrics = {
        'sharpe_ratio': calculate_sharpe_ratio(trading_returns),
        'sortino_ratio': calculate_sortino_ratio(trading_returns),
        'profit_factor': calculate_profit_factor(trading_returns),
        'win_rate': calculate_win_rate(trading_returns),
        'expectancy': calculate_expectancy(trading_returns),
        'max_drawdown': calculate_max_drawdown(cumulative_returns),
        'total_return': np.sum(trading_returns),
        'num_trades': len(trading_returns),
        'accuracy': np.mean(predictions == actual_labels),
        'trades_stopped_by_goal': trades_stopped_by_goal,
        'trades_stopped_by_loss': trades_stopped_by_loss
    }

    return metrics
