"""Pure prediction functions (wrappers around model logic)."""

from typing import Tuple, Dict, Any


def apply_confidence_threshold(
    signal: str,
    confidence: float,
    threshold: float = 0.25
) -> Tuple[str, bool]:
    """
    Apply confidence threshold filtering.

    Pure function that filters signals based on confidence.

    Args:
        signal: Original signal
        confidence: Confidence score (0-1)
        threshold: Minimum confidence threshold

    Returns:
        Tuple of (filtered_signal, was_filtered)

    Example:
        >>> apply_confidence_threshold("buy", 0.8, 0.25)
        ('buy', False)
        >>> apply_confidence_threshold("buy", 0.1, 0.25)
        ('hold', True)
    """
    if confidence < threshold:
        return "hold", True
    return signal, False


def should_block_prediction_during_training(is_training: bool, is_trained: bool) -> Tuple[bool, str]:
    """
    Determine if predictions should be blocked.

    Pure function with no side effects.

    Args:
        is_training: Whether model is currently training
        is_trained: Whether model has been trained

    Returns:
        Tuple of (should_block, reason)
    """
    if is_training:
        return True, "Model is currently training"

    if not is_trained:
        return True, "Model is not trained yet"

    return False, ""


def determine_signal_action(
    signal: str,
    confidence: float,
    exit_analysis: Dict[str, Any]
) -> str:
    """
    Determine final signal action considering exit analysis.

    Pure function that combines signal with exit logic.

    Args:
        signal: Original signal
        confidence: Confidence score
        exit_analysis: Exit analysis dictionary

    Returns:
        Final signal action
    """
    # If early exit detected, override to hold
    if exit_analysis.get('should_exit', False):
        return "hold"

    return signal


def calculate_trailing_stop_active(
    current_position: str,
    entry_price: float,
    trailing_stop_price: float
) -> bool:
    """
    Determine if trailing stop is active.

    Pure function with no side effects.

    Args:
        current_position: Current position ("flat", "long", "short")
        entry_price: Entry price of position
        trailing_stop_price: Calculated trailing stop price

    Returns:
        True if trailing stop is active
    """
    return (
        current_position in ['long', 'short']
        and entry_price > 0
        and trailing_stop_price != 0.0
    )


def format_trade_params_response(trade_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format trade parameters for response.

    Pure function that extracts and formats parameters.

    Args:
        trade_params: Trade parameters from model

    Returns:
        Formatted parameters dictionary
    """
    from core.transformations import sanitize_float

    return {
        "contracts": trade_params.get('contracts', 0),
        "entry_price": sanitize_float(trade_params.get('entry_price', 0)),
        "stop_loss": sanitize_float(trade_params.get('stop_loss', 0)),
        "take_profit": sanitize_float(trade_params.get('take_profit', 0)),
        "stop_distance": sanitize_float(trade_params.get('stop_distance', 0)),
        "target_distance": sanitize_float(trade_params.get('target_distance', 0)),
        "risk_reward_ratio": sanitize_float(trade_params.get('risk_reward', 0)),
        "risk_dollars": sanitize_float(trade_params.get('risk_dollars', 0)),
        "risk_pct": sanitize_float(trade_params.get('risk_pct', 0)),
        "regime": trade_params.get('regime', 'unknown'),
        "trend_direction": trade_params.get('trend_direction', 'neutral'),
        "trend_strength": sanitize_float(trade_params.get('trend_strength', 0)),
        "is_counter_trend": trade_params.get('is_counter_trend', False),
        "target_adjustment": sanitize_float(trade_params.get('target_adjustment', 1.0))
    }


def format_counter_trend_filter_response(trade_params: Dict[str, Any], signal: str, confidence: float) -> Dict[str, Any]:
    """
    Format counter-trend filter information for response.

    Args:
        trade_params: Trade parameters
        signal: Original signal
        confidence: Original confidence

    Returns:
        Formatted counter-trend filter dictionary
    """
    from core.transformations import sanitize_float

    filter_details = trade_params.get('filter_details', {})

    return {
        "applied": filter_details.get('filtered', False) or filter_details.get('boosted', False),
        "reason": filter_details.get('reason', 'No filtering'),
        "original_signal": trade_params.get('original_signal', signal),
        "original_confidence": sanitize_float(trade_params.get('original_confidence', confidence)),
        "confidence_adjustment": sanitize_float(
            filter_details.get('confidence_boost', 0) - filter_details.get('confidence_penalty', 0)
        )
    }


def format_exit_analysis_response(exit_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format exit analysis for response.

    Args:
        exit_analysis: Exit analysis dictionary

    Returns:
        Formatted exit analysis dictionary
    """
    return {
        "should_exit": exit_analysis.get('should_exit', False),
        "reason": exit_analysis.get('reason', 'No position'),
        "urgency": exit_analysis.get('urgency', 'none'),
        "early_exit_triggered": exit_analysis.get('should_exit', False)
    }


def format_trailing_stop_response(
    trailing_stop_price: float,
    current_position: str,
    entry_price: float
) -> Dict[str, Any]:
    """
    Format trailing stop information for response.

    Args:
        trailing_stop_price: Calculated trailing stop price
        current_position: Current position
        entry_price: Entry price

    Returns:
        Formatted trailing stop dictionary
    """
    from core.transformations import sanitize_float

    return {
        "enabled": current_position in ['long', 'short'] and entry_price > 0,
        "trailing_stop_price": sanitize_float(trailing_stop_price),
        "use_trailing": trailing_stop_price != 0.0,
        "activation_threshold": "1R profit",
        "trail_distance": "0.75 ATR"
    }
