"""Pure data transformation functions for trading data."""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import math
from datetime import datetime, timezone


def sanitize_float(value: float) -> float:
    """
    Convert inf/nan to valid JSON numbers.

    Pure function with no side effects.

    Args:
        value: Float value to sanitize

    Returns:
        Sanitized float value (0.0 if inf/nan)

    Example:
        >>> sanitize_float(5.0)
        5.0
        >>> sanitize_float(float('inf'))
        0.0
        >>> sanitize_float(float('nan'))
        0.0
    """
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def sanitize_dict_floats(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively sanitize all float values in a dictionary.
    Also converts numpy types to native Python types for JSON serialization.

    Args:
        data: Dictionary potentially containing float values or numpy types

    Returns:
        New dictionary with sanitized floats and Python native types
    """
    result = {}
    for key, value in data.items():
        # Handle numpy boolean (np.bool_ only, np.bool deprecated)
        if isinstance(value, np.bool_):
            result[key] = bool(value)
        # Handle Python boolean (must check before numpy types as numpy can also match bool)
        elif isinstance(value, bool):
            result[key] = value
        # Handle numpy integers
        elif isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            result[key] = int(value)
        # Handle numpy floats
        elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
            result[key] = sanitize_float(float(value))
        # Handle Python floats
        elif isinstance(value, float):
            result[key] = sanitize_float(value)
        # Handle nested dictionaries
        elif isinstance(value, dict):
            result[key] = sanitize_dict_floats(value)
        # Handle lists
        elif isinstance(value, list):
            result[key] = [
                sanitize_dict_floats(item) if isinstance(item, dict)
                else bool(item) if isinstance(item, np.bool_)
                else item if isinstance(item, bool)
                else int(item) if isinstance(item, (np.integer, np.int64, np.int32, np.int16, np.int8))
                else sanitize_float(float(item)) if isinstance(item, (np.floating, np.float64, np.float32, np.float16))
                else sanitize_float(item) if isinstance(item, float)
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def bar_to_dict(bar_data: Any,
                bid_volume: float = 0.0,
                ask_volume: float = 0.0,
                daily_pnl: float = 0.0,
                daily_goal: float = 0.0,
                daily_max_loss: float = 0.0,
                timeframe: str = '1m') -> Dict[str, Any]:
    """
    Convert bar data to dictionary format.

    Pure function that extracts and formats bar data.

    Args:
        bar_data: Bar data (can be dict or object with attributes)
        bid_volume: Optional bid volume
        ask_volume: Optional ask volume
        daily_pnl: Daily profit/loss
        daily_goal: Daily profit goal
        daily_max_loss: Daily maximum loss limit
        timeframe: Timeframe identifier

    Returns:
        Dictionary with standardized bar data
    """
    if isinstance(bar_data, dict):
        return {
            'time': bar_data['time'],
            'open': float(bar_data['open']),
            'high': float(bar_data['high']),
            'low': float(bar_data['low']),
            'close': float(bar_data['close']),
            'volume': float(bar_data.get('volume', 0.0)),
            'bid_volume': float(bid_volume),
            'ask_volume': float(ask_volume),
            'dailyPnL': float(daily_pnl),
            'dailyGoal': float(daily_goal),
            'dailyMaxLoss': float(daily_max_loss),
            'timeframe': timeframe
        }
    else:
        # Assume object with attributes
        return {
            'time': getattr(bar_data, 'time'),
            'open': float(getattr(bar_data, 'open')),
            'high': float(getattr(bar_data, 'high')),
            'low': float(getattr(bar_data, 'low')),
            'close': float(getattr(bar_data, 'close')),
            'volume': float(getattr(bar_data, 'volume', 0.0)),
            'bid_volume': float(bid_volume),
            'ask_volume': float(ask_volume),
            'dailyPnL': float(daily_pnl),
            'dailyGoal': float(daily_goal),
            'dailyMaxLoss': float(daily_max_loss),
            'timeframe': timeframe
        }


def bars_to_dataframe(bars: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of bar dictionaries to DataFrame.

    Pure function that creates DataFrame from bars.

    Args:
        bars: List of bar dictionaries

    Returns:
        DataFrame with bar data and datetime index

    Example:
        >>> bars = [{"time": "2024-01-01", "open": 100, "high": 105, "low": 99, "close": 102}]
        >>> df = bars_to_dataframe(bars)
        >>> 'close' in df.columns
        True
    """
    df = pd.DataFrame(bars)
    df['time'] = pd.to_datetime(df['time'])
    return df


def extract_request_type(request: Dict[str, Any]) -> str:
    """
    Extract request type (historical or realtime).

    Args:
        request: Request dictionary

    Returns:
        Request type: 'historical' or 'realtime'
    """
    if 'bars' in request or 'primary_bars' in request:
        return 'historical'
    elif 'primary_bar' in request or 'time' in request:
        return 'realtime'
    return 'unknown'


def extract_bars_from_request(request: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract primary and secondary bars from request.

    Args:
        request: Request dictionary

    Returns:
        Tuple of (primary_bars, secondary_bars)
    """
    # Get primary bars
    primary_bars = request.get('primary_bars') or request.get('bars', [])

    # Get secondary bars
    secondary_bars = request.get('secondary_bars', [])

    return primary_bars, secondary_bars


def add_timestamp(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add timestamp to response (side effect wrapper).

    Note: This function has side effects (reads current time).
    Keep it separate from pure transformations.

    Args:
        data: Response dictionary

    Returns:
        Response with timestamp added
    """
    return {
        **data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def format_prediction_response(
    signal: str,
    confidence: float,
    raw_signal: str,
    filtered: bool,
    risk_management: Optional[Dict[str, Any]] = None,
    exit_analysis: Optional[Dict[str, Any]] = None,
    trailing_stop: Optional[Dict[str, Any]] = None,
    counter_trend_filter: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format prediction results into response dictionary.

    Pure function for creating response structure.

    Args:
        signal: Predicted signal
        confidence: Confidence score
        raw_signal: Original signal before filtering
        filtered: Whether signal was filtered
        risk_management: Optional risk management parameters
        exit_analysis: Optional exit analysis
        trailing_stop: Optional trailing stop info
        counter_trend_filter: Optional counter-trend filter info

    Returns:
        Formatted response dictionary
    """
    response = {
        "signal": signal,
        "raw_signal": raw_signal,
        "confidence": sanitize_float(confidence),
        "filtered": filtered,
        "recommendation": f"{signal.upper()} with {sanitize_float(confidence)*100:.1f}% confidence" +
                         (f" (filtered from {raw_signal.upper()})" if filtered else ""),
    }

    if risk_management:
        response["risk_management"] = sanitize_dict_floats(risk_management)

    if exit_analysis:
        response["exit_analysis"] = exit_analysis

    if trailing_stop:
        response["trailing_stop"] = sanitize_dict_floats(trailing_stop)

    if counter_trend_filter:
        response["counter_trend_filter"] = sanitize_dict_floats(counter_trend_filter)

    return response


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between min and max.

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value

    Example:
        >>> clamp(5.0, 0.0, 10.0)
        5.0
        >>> clamp(-5.0, 0.0, 10.0)
        0.0
        >>> clamp(15.0, 0.0, 10.0)
        10.0
    """
    return max(min_val, min(value, max_val))
