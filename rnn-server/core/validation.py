"""Pure validation functions for trading data."""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import math


def validate_bar_data(bar: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate that a bar has required fields.

    Pure function with no side effects.

    Args:
        bar: Dictionary containing bar data

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> bar = {"time": "2024-01-01", "open": 100, "high": 105, "low": 99, "close": 102}
        >>> validate_bar_data(bar)
        (True, None)
    """
    required_fields = ['time', 'open', 'high', 'low', 'close']

    for field in required_fields:
        if field not in bar:
            return False, f"Missing required field: {field}"

    # Validate numeric fields
    numeric_fields = ['open', 'high', 'low', 'close']
    for field in numeric_fields:
        try:
            value = float(bar[field])
            if not math.isfinite(value):
                return False, f"Invalid {field}: must be finite number"
            if value <= 0:
                return False, f"Invalid {field}: must be positive"
        except (ValueError, TypeError):
            return False, f"Invalid {field}: must be numeric"

    # Validate price relationships
    high = float(bar['high'])
    low = float(bar['low'])
    open_price = float(bar['open'])
    close = float(bar['close'])

    if low > high:
        return False, "Low price cannot be greater than high price"

    if open_price < low or open_price > high:
        return False, "Open price must be between low and high"

    if close < low or close > high:
        return False, "Close price must be between low and high"

    return True, None


def validate_bars_list(bars: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """
    Validate a list of bars.

    Args:
        bars: List of bar dictionaries

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not bars:
        return False, "Bars list cannot be empty"

    if not isinstance(bars, list):
        return False, "Bars must be a list"

    # Validate each bar
    for i, bar in enumerate(bars):
        is_valid, error = validate_bar_data(bar)
        if not is_valid:
            return False, f"Bar {i}: {error}"

    return True, None


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate that a DataFrame has required columns and data.

    Pure function with no side effects.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None or len(df) == 0:
        return False, "DataFrame is empty"

    required_columns = ['time', 'open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"

    # Check for any NaN values in critical columns
    for col in required_columns:
        if df[col].isna().any():
            return False, f"Column '{col}' contains NaN values"

    return True, None


def validate_features(features: Any) -> Tuple[bool, Optional[str]]:
    """
    Validate feature data for prediction.

    Args:
        features: Feature data (DataFrame, array, etc.)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if features is None:
        return False, "Features cannot be None"

    if isinstance(features, pd.DataFrame):
        if len(features) == 0:
            return False, "Features DataFrame is empty"
        return True, None

    return True, None


def validate_request_fields(request: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate that request has required fields.

    Args:
        request: Request dictionary
        required_fields: List of required field names

    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_fields = [field for field in required_fields if field not in request]

    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"

    return True, None


def validate_confidence(confidence: float) -> Tuple[bool, Optional[str]]:
    """
    Validate confidence score is in valid range [0, 1].

    Args:
        confidence: Confidence score

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not math.isfinite(confidence):
        return False, "Confidence must be finite"

    if confidence < 0.0 or confidence > 1.0:
        return False, "Confidence must be between 0 and 1"

    return True, None


def validate_account_balance(balance: float) -> Tuple[bool, Optional[str]]:
    """
    Validate account balance is positive.

    Args:
        balance: Account balance

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not math.isfinite(balance):
        return False, "Account balance must be finite"

    if balance <= 0:
        return False, "Account balance must be positive"

    return True, None
