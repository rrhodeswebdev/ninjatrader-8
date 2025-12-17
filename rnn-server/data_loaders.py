"""
Unified Data Loading Utilities

This module provides data loading functions that work with both:
- RNN server backtesting (event-driven)
- backintime framework (declarative strategies)

It handles format conversions and ensures data compatibility across both systems.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
from datetime import datetime, timedelta
from trading_metrics import (
    calculate_expectancy,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)


class DataLoader:
    """Unified data loader for both backtesting frameworks"""

    @staticmethod
    def load_csv(
        file_path: str,
        time_col: str = 'time',
        date_format: Optional[str] = None,
        timezone: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file. Handles multiple formats automatically:

        Format 1: timestamp,open,high,low,close,volume,close_time
        Format 2: Timestamp,Open,High,Low,Close,Volume

        Args:
            file_path: Path to CSV file
            time_col: Name of time column (default: 'time', auto-detected)
            date_format: Optional strftime format for parsing dates
            timezone: Optional timezone to localize timestamps

        Returns:
            DataFrame with standardized format:
            - time (datetime64[ns])
            - open, high, low, close, volume (float64)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Read CSV
        df = pd.read_csv(file_path)

        # Normalize column names to lowercase for case-insensitive handling
        df.columns = df.columns.str.lower().str.strip()

        # Auto-detect time column name (support 'time', 'timestamp', etc.)
        possible_time_cols = ['time', 'timestamp', 'datetime', 'date']
        time_col_found = None
        for col in possible_time_cols:
            if col in df.columns:
                time_col_found = col
                break

        if time_col_found is None:
            raise ValueError(f"Could not find time column. Available columns: {df.columns.tolist()}")

        # Parse time column with automatic format detection
        if date_format:
            df['time'] = pd.to_datetime(df[time_col_found], format=date_format)
        else:
            # Let pandas infer the format automatically (handles both YYYY-MM-DD and MM/DD/YYYY)
            df['time'] = pd.to_datetime(df[time_col_found], infer_datetime_format=True)

        # Localize timezone if specified
        if timezone:
            if df['time'].dt.tz is None:
                df['time'] = df['time'].dt.tz_localize(timezone)
            else:
                df['time'] = df['time'].dt.tz_convert(timezone)

        # Ensure required columns exist (now lowercase)
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Available: {df.columns.tolist()}")

        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 0.0

        # Standardize column order (ignore close_time if present)
        standard_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        df = df[standard_cols].copy()

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove any NaN rows
        df = df.dropna()

        print(f" Loaded {len(df)} bars from {file_path.name}")
        print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
        print(f"  Price range: ${df['low'].min():.2f} to ${df['high'].max():.2f}")

        return df

    @staticmethod
    def load_ninjatrader_csv(
        file_path: str,
        timezone: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load NinjaTrader export data from CSV file.

        NinjaTrader format: YYYYMMDD HHMMSS;open;high;low;close;volume (no header)

        Args:
            file_path: Path to NinjaTrader CSV file
            timezone: Optional timezone to localize timestamps

        Returns:
            DataFrame with standardized format
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Read CSV with semicolon delimiter
        df = pd.read_csv(
            file_path,
            sep=';',
            header=None,
            names=['datetime', 'open', 'high', 'low', 'close', 'volume']
        )

        # Parse datetime (format: YYYYMMDD HHMMSS)
        df['time'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S')
        df = df.drop(columns=['datetime'])

        # Localize timezone if specified
        if timezone:
            if df['time'].dt.tz is None:
                df['time'] = df['time'].dt.tz_localize(timezone)
            else:
                df['time'] = df['time'].dt.tz_convert(timezone)

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove any NaN rows
        df = df.dropna()

        # Standardize column order
        standard_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        df = df[standard_cols].copy()

        print(f" Loaded {len(df)} bars from {file_path.name} (NinjaTrader format)")
        print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
        print(f"  Price range: ${df['low'].min():.2f} to ${df['high'].max():.2f}")

        return df

    @staticmethod
    def save_for_rnn_backtester(
        df: pd.DataFrame,
        output_file: str,
        include_orderflow: bool = False
    ) -> str:
        """
        Save DataFrame in format for RNN backtester.

        Format: time,open,high,low,close,volume[,bid_volume,ask_volume]

        Args:
            df: DataFrame with OHLCV data
            output_file: Output file path
            include_orderflow: Include bid/ask volume columns (default: False)

        Returns:
            Path to saved file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare columns
        cols = ['time', 'open', 'high', 'low', 'close', 'volume']

        if include_orderflow:
            if 'bid_volume' not in df.columns:
                df['bid_volume'] = df['volume'] * 0.5  # Estimate if not available
            if 'ask_volume' not in df.columns:
                df['ask_volume'] = df['volume'] * 0.5
            cols.extend(['bid_volume', 'ask_volume'])

        # Format time as string
        df_out = df[cols].copy()
        df_out['time'] = df_out['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Save
        df_out.to_csv(output_path, index=False)

        print(f" Saved RNN format data to {output_path}")
        print(f"  {len(df_out)} bars")

        return str(output_path)

    @staticmethod
    def save_for_backintime(
        df: pd.DataFrame,
        output_file: str,
        bar_duration: timedelta = timedelta(minutes=1)
    ) -> str:
        """
        Save DataFrame in format for backintime framework.

        Format: open_time,open,high,low,close,volume,close_time (no header)

        Args:
            df: DataFrame with OHLCV data
            output_file: Output file path
            bar_duration: Duration of each bar (default: 1 minute)

        Returns:
            Path to saved file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate close time
        df_out = df.copy()
        df_out['close_time'] = df_out['time'] + bar_duration

        # Format timestamps
        df_out['time'] = df_out['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_out['close_time'] = df_out['close_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Column order for backintime
        cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'close_time']
        df_out = df_out[cols]

        # Save without header
        df_out.to_csv(output_path, index=False, header=False)

        print(f" Saved backintime format data to {output_path}")
        print(f"  {len(df_out)} bars")

        return str(output_path)

    @staticmethod
    def split_train_test(
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets.

        Args:
            df: Full DataFrame
            train_ratio: Fraction for training (default: 0.7)
            validation_ratio: Fraction for validation (default: 0.15)
            Remaining data goes to test set

        Returns:
            Tuple of (train_df, validation_df, test_df)
        """
        total_len = len(df)
        train_size = int(total_len * train_ratio)
        val_size = int(total_len * validation_ratio)

        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size + val_size].copy()
        test_df = df.iloc[train_size + val_size:].copy()

        print(f"\n Data Split:")
        print(f"  Training:   {len(train_df):5d} bars ({train_df['time'].min()} to {train_df['time'].max()})")
        print(f"  Validation: {len(val_df):5d} bars ({val_df['time'].min()} to {val_df['time'].max()})")
        print(f"  Test:       {len(test_df):5d} bars ({test_df['time'].min()} to {test_df['time'].max()})")

        return train_df, val_df, test_df

    @staticmethod
    def filter_trading_hours(
        df: pd.DataFrame,
        start_time: str = "09:30",
        end_time: str = "16:00",
        exclude_weekends: bool = True
    ) -> pd.DataFrame:
        """
        Filter data to regular trading hours.

        Args:
            df: DataFrame with time column
            start_time: Session start time (HH:MM format)
            end_time: Session end time (HH:MM format)
            exclude_weekends: Remove Saturday/Sunday data (default: True)

        Returns:
            Filtered DataFrame
        """
        df_filtered = df.copy()

        # Parse times
        start_hour, start_min = map(int, start_time.split(':'))
        end_hour, end_min = map(int, end_time.split(':'))

        # Filter by time of day
        mask = (
            (df_filtered['time'].dt.hour > start_hour) |
            ((df_filtered['time'].dt.hour == start_hour) & (df_filtered['time'].dt.minute >= start_min))
        ) & (
            (df_filtered['time'].dt.hour < end_hour) |
            ((df_filtered['time'].dt.hour == end_hour) & (df_filtered['time'].dt.minute <= end_min))
        )

        df_filtered = df_filtered[mask]

        # Exclude weekends if requested
        if exclude_weekends:
            df_filtered = df_filtered[df_filtered['time'].dt.dayofweek < 5]

        print(f" Filtered to trading hours ({start_time}-{end_time})")
        print(f"  {len(df_filtered)} bars remaining (from {len(df)})")

        return df_filtered

    @staticmethod
    def resample_timeframe(
        df: pd.DataFrame,
        target_timeframe: str = '5min'
    ) -> pd.DataFrame:
        """
        Resample data to a different timeframe.

        Args:
            df: DataFrame with 1-minute bars
            target_timeframe: Target timeframe (e.g., '5min', '15min', '1H')

        Returns:
            Resampled DataFrame
        """
        df_resampled = df.set_index('time').resample(target_timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()

        print(f" Resampled to {target_timeframe}")
        print(f"  {len(df_resampled)} bars (from {len(df)})")

        return df_resampled


def convert_between_formats(
    input_file: str,
    output_file: str,
    output_format: str = 'rnn',
    **kwargs
) -> str:
    """
    Convenience function to convert between formats.

    Args:
        input_file: Input CSV file
        output_file: Output CSV file
        output_format: Target format ('rnn' or 'backintime')
        **kwargs: Additional arguments for load_csv() and save functions

    Returns:
        Path to output file
    """
    loader = DataLoader()

    # Load data
    df = loader.load_csv(input_file, **kwargs)

    # Save in target format
    if output_format.lower() == 'rnn':
        return loader.save_for_rnn_backtester(df, output_file)
    elif output_format.lower() == 'backintime':
        return loader.save_for_backintime(df, output_file)
    else:
        raise ValueError(f"Unknown format: {output_format}. Use 'rnn' or 'backintime'")


def compare_backtest_results(
    rnn_results: dict,
    backintime_results,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare results from RNN backtester and backintime framework.

    This function analyzes differences between both approaches to help understand
    the impact of realistic execution simulation.

    Args:
        rnn_results: Results dict from RNN Backtester
        backintime_results: Results object from backintime framework
        verbose: Print detailed comparison (default: True)

    Returns:
        DataFrame with side-by-side comparison of key metrics
    """
    print("\n" + "="*70)
    print("BACKTEST RESULTS COMPARISON")
    print("="*70)

    # Extract backintime stats
    try:
        bt_stats = backintime_results.get_stats()
    except:
        bt_stats = {}

    # Build comparison DataFrame
    metrics = []

    def compute_max_dd_from_equity(equity_curve):
        """Compute max drawdown (%) from an equity curve safely."""
        try:
            eq = np.array(equity_curve, dtype=float)
        except Exception:
            return 0.0
        if eq.size == 0:
            return 0.0
        start_cap = eq[0]
        if start_cap <= 0:
            return 0.0
        # Normalize to start at 1.0 to avoid divide-by-zero blowups
        normalized = (eq / start_cap).astype(float)
        normalized = np.insert(normalized, 0, 1.0)
        return calculate_max_drawdown(normalized) * 100

    def compute_rnn_metrics(rnn_result: dict):
        """Lightweight recompute for RNN backtester to avoid pathological values."""
        derived = {}
        if not isinstance(rnn_result, dict):
            return derived

        try:
            equity_curve = np.array(rnn_result.get('equity_curve', []), dtype=float)
        except Exception:
            equity_curve = np.array([])

        if equity_curve.size:
            start_cap = equity_curve[0]
            derived['max_drawdown'] = compute_max_dd_from_equity(equity_curve)
            derived['total_return_pct'] = ((equity_curve[-1] - start_cap) / start_cap) * 100 if start_cap else rnn_result.get('total_return_pct', 0)

        return derived

    rnn_derived = compute_rnn_metrics(rnn_results)

    def to_float(val, default=0.0):
        """Normalize Decimals/TradeProfit/None -> float."""
        if val is None:
            return default
        try:
            return float(val)
        except Exception:
            return default

    # Helper to safely get values from dict or object
    def safe_get(d, key, default=0):
        if d is None:
            return default
        # If it's a dict, use .get()
        if hasattr(d, 'get'):
            val = d.get(key, default)
        # Otherwise try getattr for objects
        else:
            val = getattr(d, key, default)
        # Normalize Decimal to float to avoid mixing types with floats
        try:
            from decimal import Decimal
            if isinstance(val, Decimal):
                return float(val)
        except Exception:
            pass
        try:
            from backintime.result.futures_stats import TradeProfit
            if isinstance(val, TradeProfit):
                return float(val)
        except Exception:
            pass
        return val

    def compute_backintime_metrics(bt_result):
        """Derive richer metrics from backintime BacktestingResult."""
        derived = {}
        if bt_result is None:
            return derived

        # Try to pull stats/trades from the result object
        try:
            bt_stats_obj = bt_result.get_stats()
        except Exception:
            bt_stats_obj = None

        # Trades sequence (private attr or accessor)
        trades_seq = []
        for attr in ('_trades', 'trades', 'get_trades'):
            try:
                candidate = getattr(bt_result, attr)
                trades_seq = candidate() if callable(candidate) else candidate
                if trades_seq:
                    break
            except Exception:
                continue

        # Compute per-trade profits using backintime helper if available
        profits = []
        try:
            from backintime.result.futures_stats import get_trades_profit
            profits = get_trades_profit(trades_seq) if trades_seq else []
        except Exception:
            profits = []

        profit_vals = np.array([to_float(p.absolute_profit, 0.0) for p in profits], dtype=float) if profits else np.array([])

        start_balance = to_float(getattr(bt_result, 'start_balance', None), 0.0)
        result_equity = to_float(getattr(bt_result, 'result_equity', None), 0.0)
        if not result_equity and start_balance and profit_vals.size:
            result_equity = start_balance + profit_vals.sum()

        stats_trades = to_float(getattr(bt_stats_obj, 'trades_count', 0), 0)
        result_trades = to_float(getattr(bt_result, 'trades_count', 0), 0)
        profit_trades = profit_vals.size
        derived['total_trades'] = int(max(profit_trades, result_trades, stats_trades))

        wins_from_profits = int((profit_vals > 0).sum()) if profit_vals.size else 0
        stats_wins = to_float(getattr(bt_stats_obj, 'wins_count', 0), 0)
        derived['winning_trades'] = int(max(wins_from_profits, stats_wins))

        losses_from_profits = int((profit_vals < 0).sum()) if profit_vals.size else 0
        stats_losses = to_float(getattr(bt_stats_obj, 'losses_count', 0), 0)
        derived['losing_trades'] = int(max(losses_from_profits, stats_losses))

        derived['total_pnl'] = profit_vals.sum() if profit_vals.size else to_float(getattr(bt_result, 'total_gain', 0), 0)
        derived['avg_trade_pnl'] = profit_vals.mean() if profit_vals.size else to_float(getattr(bt_stats_obj, 'average_profit_all', 0), 0)
        derived['avg_win'] = profit_vals[profit_vals > 0].mean() if (profit_vals > 0).any() else to_float(getattr(bt_stats_obj, 'average_profit', 0), 0)
        derived['avg_loss'] = profit_vals[profit_vals < 0].mean() if (profit_vals < 0).any() else to_float(getattr(bt_stats_obj, 'average_loss', 0), 0)
        derived['largest_win'] = profit_vals.max() if profit_vals.size else to_float(getattr(bt_stats_obj, 'best_deal_absolute', 0), 0)
        derived['largest_loss'] = profit_vals.min() if profit_vals.size else to_float(getattr(bt_stats_obj, 'worst_deal_absolute', 0), 0)
        derived['expectancy'] = calculate_expectancy(profit_vals) if profit_vals.size else to_float(getattr(bt_stats_obj, 'expectancy', 0), 0)

        # Risk metrics using reconstructed equity curve where possible
        equity_curve = None
        if profit_vals.size and start_balance:
            equity_curve = start_balance + np.cumsum(profit_vals)

        if equity_curve is not None and equity_curve.size:
            derived['max_drawdown'] = compute_max_dd_from_equity(equity_curve)
        else:
            derived['max_drawdown'] = to_float(getattr(bt_stats_obj, 'max_drawdown', 0), 0)

        returns_pct = profit_vals / start_balance if start_balance else profit_vals
        derived['sharpe_ratio'] = calculate_sharpe_ratio(returns_pct, periods_per_year=252) if returns_pct.size else 0
        derived['sortino_ratio'] = calculate_sortino_ratio(returns_pct, periods_per_year=252) if returns_pct.size else 0
        derived['profit_factor'] = calculate_profit_factor(profit_vals) if profit_vals.size else to_float(getattr(bt_stats_obj, 'profit_factor', 0), 0)

        if start_balance:
            derived['total_return_pct'] = ((result_equity - start_balance) / start_balance) * 100
        else:
            derived['total_return_pct'] = to_float(getattr(bt_result, 'total_gain_percents', 0), 0)

        # backintime win_rate is already percent
        if bt_stats_obj is not None:
            try:
                derived['win_rate'] = float(bt_stats_obj.win_rate)
            except Exception:
                derived['win_rate'] = 0.0
        else:
            derived['win_rate'] = 0.0

        # Fallback: compute win rate from derived counts if stats were missing
        if derived['win_rate'] == 0.0 and derived['total_trades'] > 0:
            derived['win_rate'] = (derived['winning_trades'] / derived['total_trades']) * 100

        return derived

    # Trade Statistics
    bt_derived = compute_backintime_metrics(backintime_results)

    metrics.append({
        'Metric': 'Total Trades',
        'RNN Backtester': safe_get(rnn_results, 'total_trades', 0),
        'backintime': bt_derived.get('total_trades', safe_get(bt_stats, 'total_trades', 0)),
        'Difference': 0,
        'Unit': 'trades'
    })

    def normalize_win_rate(val):
        try:
            v = float(val)
        except Exception:
            return 0.0
        # backintime win_rate already 0-100; rnn win_rate is 0-1
        return v if v > 1 else v * 100

    metrics.append({
        'Metric': 'Win Rate',
        'RNN Backtester': normalize_win_rate(safe_get(rnn_results, 'win_rate', 0)),
        'backintime': normalize_win_rate(bt_derived.get('win_rate', safe_get(bt_stats, 'win_rate', 0))),
        'Difference': normalize_win_rate(safe_get(rnn_results, 'win_rate', 0)) - normalize_win_rate(bt_derived.get('win_rate', safe_get(bt_stats, 'win_rate', 0))),
        'Unit': '%'
    })

    metrics.append({
        'Metric': 'Winning Trades',
        'RNN Backtester': safe_get(rnn_results, 'winning_trades', 0),
        'backintime': bt_derived.get('winning_trades', safe_get(bt_stats, 'wins_count', 0)),
        'Difference': 0,
        'Unit': 'trades'
    })

    metrics.append({
        'Metric': 'Losing Trades',
        'RNN Backtester': safe_get(rnn_results, 'losing_trades', 0),
        'backintime': bt_derived.get('losing_trades', safe_get(bt_stats, 'losses_count', 0)),
        'Difference': 0,
        'Unit': 'trades'
    })

    # P&L Metrics
    metrics.append({
        'Metric': 'Total P&L',
        'RNN Backtester': safe_get(rnn_results, 'total_pnl', 0),
        'backintime': bt_derived.get('total_pnl', safe_get(bt_stats, 'total_pnl', 0)),
        'Difference': 0,
        'Unit': '$'
    })

    metrics.append({
        'Metric': 'Total Return',
        'RNN Backtester': rnn_derived.get('total_return_pct', safe_get(rnn_results, 'total_return_pct', 0)),
        'backintime': bt_derived.get('total_return_pct', safe_get(bt_stats, 'total_return_pct', 0)),
        'Difference': 0,
        'Unit': '%'
    })

    # Risk-Adjusted Metrics
    metrics.append({
        'Metric': 'Sharpe Ratio',
        'RNN Backtester': safe_get(rnn_results, 'sharpe_ratio', 0),
        'backintime': bt_derived.get('sharpe_ratio', safe_get(bt_stats, 'sharpe_ratio', 0)),
        'Difference': 0,
        'Unit': 'ratio'
    })

    metrics.append({
        'Metric': 'Sortino Ratio',
        'RNN Backtester': safe_get(rnn_results, 'sortino_ratio', 0),
        'backintime': bt_derived.get('sortino_ratio', safe_get(bt_stats, 'sortino_ratio', 0)),
        'Difference': 0,
        'Unit': 'ratio'
    })

    metrics.append({
        'Metric': 'Profit Factor',
        'RNN Backtester': safe_get(rnn_results, 'profit_factor', 0),
        'backintime': bt_derived.get('profit_factor', safe_get(bt_stats, 'profit_factor', 0)),
        'Difference': 0,
        'Unit': 'ratio'
    })

    metrics.append({
        'Metric': 'Max Drawdown',
        'RNN Backtester': rnn_derived.get('max_drawdown', safe_get(rnn_results, 'max_drawdown', 0)),
        'backintime': bt_derived.get('max_drawdown', safe_get(bt_stats, 'max_drawdown', 0)),
        'Difference': 0,
        'Unit': '%'
    })

    # Trade Metrics
    metrics.append({
        'Metric': 'Avg Trade P&L',
        'RNN Backtester': safe_get(rnn_results, 'avg_trade_pnl', 0),
        'backintime': bt_derived.get('avg_trade_pnl', safe_get(bt_stats, 'avg_trade_pnl', 0)),
        'Difference': 0,
        'Unit': '$'
    })

    metrics.append({
        'Metric': 'Avg Win',
        'RNN Backtester': safe_get(rnn_results, 'avg_win', 0),
        'backintime': bt_derived.get('avg_win', safe_get(bt_stats, 'average_profit', 0)),
        'Difference': 0,
        'Unit': '$'
    })

    metrics.append({
        'Metric': 'Avg Loss',
        'RNN Backtester': safe_get(rnn_results, 'avg_loss', 0),
        'backintime': bt_derived.get('avg_loss', safe_get(bt_stats, 'average_loss', 0)),
        'Difference': 0,
        'Unit': '$'
    })

    metrics.append({
        'Metric': 'Largest Win',
        'RNN Backtester': safe_get(rnn_results, 'largest_win', 0),
        'backintime': bt_derived.get('largest_win', safe_get(bt_stats, 'largest_win', 0)),
        'Difference': 0,
        'Unit': '$'
    })

    metrics.append({
        'Metric': 'Largest Loss',
        'RNN Backtester': safe_get(rnn_results, 'largest_loss', 0),
        'backintime': bt_derived.get('largest_loss', safe_get(bt_stats, 'largest_loss', 0)),
        'Difference': 0,
        'Unit': '$'
    })

    metrics.append({
        'Metric': 'Expectancy',
        'RNN Backtester': safe_get(rnn_results, 'expectancy', 0),
        'backintime': bt_derived.get('expectancy', safe_get(bt_stats, 'expectancy', 0)),
        'Difference': 0,
        'Unit': '$'
    })

    # Create DataFrame
    df_comparison = pd.DataFrame(metrics)

    # Simple absolute differences (no percent scaling to avoid blow-ups)
    df_comparison['Difference'] = df_comparison['backintime'].astype(float) - df_comparison['RNN Backtester'].astype(float)

    if verbose:
        print("\n Metric Comparison:")
        fmt = {
            'RNN Backtester': lambda x: f"{x:,.2f}",
            'backintime': lambda x: f"{x:,.2f}",
            'Difference': lambda x: f"{x:,.2f}"
        }
        print(df_comparison.to_string(index=False, formatters=fmt))

        print("\n" + "="*70)
        print("EXECUTION IMPACT ANALYSIS")
        print("="*70)

        # Analyze key differences
        bt_pnl = bt_derived.get('total_pnl', safe_get(bt_stats, 'total_pnl', 0))
        total_pnl_diff = safe_get(rnn_results, 'total_pnl', 0) - bt_pnl

        print(f"\n P&L Impact:")
        print(f"  RNN Backtester P&L:  ${safe_get(rnn_results, 'total_pnl', 0):>10,.2f}")
        print(f"  backintime P&L:      ${bt_pnl:>10,.2f}")
        print(f"  Difference:          ${total_pnl_diff:>10,.2f}")

        if total_pnl_diff > 0:
            print(f"\n    backintime shows LOWER P&L (more realistic)")
            print(f"     Impact of realistic execution: ${abs(total_pnl_diff):.2f}")
        elif total_pnl_diff < 0:
            print(f"\n    backintime shows HIGHER P&L")
            print(f"     Possible better fill prices: ${abs(total_pnl_diff):.2f}")

        print(f"\n Risk Metrics:")
        sharpe_rnn = safe_get(rnn_results, 'sharpe_ratio', 0)
        sharpe_bt = bt_derived.get('sharpe_ratio', safe_get(bt_stats, 'sharpe_ratio', 0))
        print(f"  RNN Sharpe:     {sharpe_rnn:>8.2f}")
        print(f"  backintime Sharpe: {sharpe_bt:>8.2f}")

        if sharpe_bt < sharpe_rnn:
            print(f"    Lower Sharpe in backintime (realistic execution impact)")
        else:
            print(f"   Similar or better Sharpe in backintime")

        print("\n" + "="*70)
        print("KEY INSIGHTS")
        print("="*70)

        insights = []

        # Trade count difference
        trades_diff = abs(safe_get(rnn_results, 'total_trades', 0) - bt_derived.get('total_trades', safe_get(bt_stats, 'total_trades', 0)))
        if trades_diff > 0:
            insights.append(f"   Trade count differs by {trades_diff} - may be due to margin constraints or session handling")

        # Performance degradation
        if total_pnl_diff > 100:
            insights.append(f"   Significant P&L impact (${total_pnl_diff:.2f}) from realistic execution")

        # Sharpe impact
        sharpe_diff = abs(sharpe_rnn - sharpe_bt)
        if sharpe_diff > 0.3:
            insights.append(f"   Sharpe ratio differs significantly ({sharpe_diff:.2f}) - execution quality matters")

        # Win rate impact
        wr_rnn = normalize_win_rate(safe_get(rnn_results, 'win_rate', 0))
        wr_bt = normalize_win_rate(bt_derived.get('win_rate', safe_get(bt_stats, 'win_rate', 0)))
        wr_diff = abs(wr_rnn - wr_bt)
        if wr_diff > 5:
            insights.append(f"   Win rate differs by {wr_diff:.1f}% - realistic fills affect outcomes")

        if insights:
            print("\n".join(insights))
        else:
            print("   Results are similar between both approaches")
            print("   Good agreement suggests robust strategy")

        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)

        bt_sharpe_val = bt_derived.get('sharpe_ratio', safe_get(bt_stats, 'sharpe_ratio', 0))
        if bt_sharpe_val > 1.0:
            print("   backintime results look strong - strategy validated")
            print("      Consider live paper trading")
        elif bt_sharpe_val > 0.5:
            print("    backintime results are marginal")
            print("      Optimize risk parameters before live trading")
        else:
            print("   backintime results are weak")
            print("      Strategy needs improvement before deployment")

        if total_pnl_diff > safe_get(rnn_results, 'total_pnl', 0) * 0.3:
            print("\n    Large execution impact (>30%)")
            print("      Focus on entry timing and limit orders")
            print("      Review slippage assumptions")

        print("\n" + "="*70)

    return df_comparison


if __name__ == '__main__':
    print(__doc__)
    print("\nExample usage:")
    print("""
    from data_loaders import DataLoader, compare_backtest_results

    # Load data
    loader = DataLoader()
    df = loader.load_csv('data.csv')

    # Split for training
    train_df, val_df, test_df = loader.split_train_test(df)

    # Filter to trading hours
    df_rth = loader.filter_trading_hours(df, "09:30", "16:00")

    # Save in different formats
    loader.save_for_rnn_backtester(train_df, 'rnn_train_data.csv')
    loader.save_for_backintime(test_df, 'backintime_test_data.csv')

    # Resample to 5-minute bars
    df_5min = loader.resample_timeframe(df, '5min')

    # Compare backtest results
    comparison = compare_backtest_results(rnn_results, backintime_results)
    """)
