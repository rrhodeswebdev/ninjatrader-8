"""
RTH (Regular Trading Hours) Filter

Filters market data to include only Regular Trading Hours for ES futures.
RTH for ES: 9:30 AM - 4:00 PM Eastern Time (Monday-Friday)
"""

import pandas as pd
from datetime import time


def filter_rth_data(df: pd.DataFrame, time_column: str = 'time') -> pd.DataFrame:
    """
    Filter DataFrame to include only Regular Trading Hours (RTH) data.

    RTH for ES futures: 9:30 AM - 4:00 PM Eastern Time

    Args:
        df: DataFrame with OHLCV data
        time_column: Name of the datetime column (default: 'time')

    Returns:
        DataFrame filtered to RTH hours only
    """
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        df[time_column] = pd.to_datetime(df[time_column])

    # Convert to Eastern Time if needed
    # If your data is already in ET, you can skip this
    if df[time_column].dt.tz is None:
        # Assume data is in ET if no timezone info
        df_copy = df.copy()
        df_copy[time_column] = df_copy[time_column].dt.tz_localize('US/Eastern', ambiguous='infer', nonexistent='shift_forward')
    else:
        df_copy = df.copy()
        df_copy[time_column] = df_copy[time_column].dt.tz_convert('US/Eastern')

    # RTH hours: 9:30 AM - 4:00 PM ET
    rth_start = time(9, 30)
    rth_end = time(16, 0)

    # Filter to RTH hours
    mask = (
        (df_copy[time_column].dt.time >= rth_start) &
        (df_copy[time_column].dt.time < rth_end) &
        (df_copy[time_column].dt.dayofweek < 5)  # Monday=0 to Friday=4
    )

    df_rth = df[mask].copy()

    # Print statistics
    original_bars = len(df)
    rth_bars = len(df_rth)
    rth_percentage = (rth_bars / original_bars * 100) if original_bars > 0 else 0

    print(f"\n RTH Filter Statistics:")
    print(f"   Original bars: {original_bars:,}")
    print(f"   RTH bars: {rth_bars:,}")
    print(f"   RTH percentage: {rth_percentage:.1f}%")
    print(f"   Filtered out: {original_bars - rth_bars:,} bars")

    if len(df_rth) > 0:
        print(f"   RTH date range: {df_rth[time_column].min()} to {df_rth[time_column].max()}")

    return df_rth


def add_session_label(df: pd.DataFrame, time_column: str = 'time') -> pd.DataFrame:
    """
    Add a column indicating the trading session (RTH vs ETH).

    This is useful if you want to keep all data but analyze sessions separately.

    Args:
        df: DataFrame with OHLCV data
        time_column: Name of the datetime column

    Returns:
        DataFrame with added 'session' column ('RTH' or 'ETH')
    """
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        df[time_column] = pd.to_datetime(df[time_column])

    # Convert to Eastern Time if needed
    if df[time_column].dt.tz is None:
        df[time_column] = df[time_column].dt.tz_localize('US/Eastern', ambiguous='infer', nonexistent='shift_forward')
    else:
        df[time_column] = df[time_column].dt.tz_convert('US/Eastern')

    # RTH hours: 9:30 AM - 4:00 PM ET
    rth_start = time(9, 30)
    rth_end = time(16, 0)

    # Create session label
    mask_rth = (
        (df[time_column].dt.time >= rth_start) &
        (df[time_column].dt.time < rth_end) &
        (df[time_column].dt.dayofweek < 5)
    )

    df['session'] = 'ETH'
    df.loc[mask_rth, 'session'] = 'RTH'

    # Print statistics
    rth_count = (df['session'] == 'RTH').sum()
    eth_count = (df['session'] == 'ETH').sum()

    print(f"\n Session Distribution:")
    print(f"   RTH bars: {rth_count:,} ({rth_count/len(df)*100:.1f}%)")
    print(f"   ETH bars: {eth_count:,} ({eth_count/len(df)*100:.1f}%)")

    return df


def get_rth_hours_info():
    """
    Print information about RTH hours for ES futures.
    """
    info = """
    
               ES E-mini S&P 500 Futures Trading Hours            
    
     RTH (Regular Trading Hours):                                 
        9:30 AM - 4:00 PM Eastern Time                          
        6:30 AM - 1:00 PM Pacific Time                          
        Monday - Friday                                          
        6.5 hours per day                                        
                                                                  
     ETH (Extended/Electronic Trading Hours):                     
        Sunday 6:00 PM - Monday 9:30 AM ET (open)               
        Monday 4:00 PM - Tuesday 9:30 AM ET (overnight)         
        ... continues through the week                           
        Friday 4:00 PM - Sunday 6:00 PM (weekend close)         
                                                                  
     RTH Characteristics:                                         
        Higher volume & liquidity                               
        Tighter bid-ask spreads                                 
        More institutional participation                         
        Economic news releases (8:30 AM, 10:00 AM ET typical)  
        More predictable price action                           
                                                                  
     ETH Characteristics:                                         
        Lower volume                                             
        Wider spreads                                            
        More retail participation                                
        Gaps at session opens                                    
        Can be more erratic                                      
    
    """
    print(info)


if __name__ == '__main__':
    # Example usage
    get_rth_hours_info()

    # Test with sample data
    import numpy as np

    print("\n" + "="*70)
    print("Testing RTH Filter with Sample Data")
    print("="*70)

    # Generate sample data spanning 3 days with 1-minute bars
    start_time = pd.Timestamp('2025-01-20 00:00:00', tz='US/Eastern')
    times = pd.date_range(start=start_time, periods=3*24*60, freq='1min')

    df_test = pd.DataFrame({
        'time': times,
        'open': 4500 + np.random.randn(len(times)) * 5,
        'high': 4505 + np.random.randn(len(times)) * 5,
        'low': 4495 + np.random.randn(len(times)) * 5,
        'close': 4500 + np.random.randn(len(times)) * 5,
        'volume': np.random.randint(100, 1000, len(times))
    })

    print(f"\nOriginal data: {len(df_test)} bars")

    # Filter to RTH only
    df_rth = filter_rth_data(df_test)

    # Add session labels
    df_labeled = add_session_label(df_test.copy())

    print("\n RTH filter test complete")
