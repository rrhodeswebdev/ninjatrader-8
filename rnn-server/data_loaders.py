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
        Load OHLCV data from CSV file.

        Args:
            file_path: Path to CSV file
            time_col: Name of time column (default: 'time')
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

        # Parse time column
        if date_format:
            df['time'] = pd.to_datetime(df[time_col], format=date_format)
        else:
            df['time'] = pd.to_datetime(df[time_col])

        # Localize timezone if specified
        if timezone:
            if df['time'].dt.tz is None:
                df['time'] = df['time'].dt.tz_localize(timezone)
            else:
                df['time'] = df['time'].dt.tz_convert(timezone)

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 0.0

        # Standardize column order
        standard_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        df = df[standard_cols].copy()

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove any NaN rows
        df = df.dropna()

        print(f"✓ Loaded {len(df)} bars from {file_path.name}")
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

        print(f"✓ Saved RNN format data to {output_path}")
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

        print(f"✓ Saved backintime format data to {output_path}")
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

        print(f"\n📊 Data Split:")
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

        print(f"✓ Filtered to trading hours ({start_time}-{end_time})")
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

        print(f"✓ Resampled to {target_timeframe}")
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


if __name__ == '__main__':
    print(__doc__)
    print("\nExample usage:")
    print("""
    from data_loaders import DataLoader

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
    """)
