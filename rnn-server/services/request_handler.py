"""Service functions for handling prediction requests using functional composition."""

from typing import Dict, Any, Callable
import pandas as pd
import numpy as np
from core.validation import (
    validate_bars_list,
    validate_dataframe,
    validate_account_balance,
)
from core.transformations import (
    bars_to_dataframe,
    format_prediction_response,
    sanitize_float,
)
from core.prediction import (
    apply_confidence_threshold,
    should_block_prediction_during_training,
    determine_signal_action,
    format_trade_params_response,
    format_counter_trend_filter_response,
    format_exit_analysis_response,
    format_trailing_stop_response,
)
from core.signal_stability import (
    get_signal_state,
    check_signal_stability,
    update_signal_state,
)
from core.market_regime import (
    calculate_market_regime,
    get_adjusted_confidence_threshold,
    should_skip_trading,
    get_regime_description,
)
from core.multi_timeframe_filter import (
    check_multi_timeframe_alignment,
    get_mtf_filter,
)


def handle_historical_request(
    bars_primary: list,
    bars_secondary: list,
    request: Dict[str, Any],
    update_model_fn: Callable,
    schedule_training_fn: Callable
) -> Dict[str, Any]:
    """
    Handle historical data request functionally.

    This function composes validation, transformation, and model update operations.

    Args:
        bars_primary: Primary timeframe bars
        bars_secondary: Secondary timeframe bars
        request: Full request dictionary
        update_model_fn: Function to update model with data
        schedule_training_fn: Function to schedule training

    Returns:
        Response dictionary
    """
    # Validate input
    is_valid, error = validate_bars_list(bars_primary)
    if not is_valid:
        return {
            "status": "error",
            "message": error
        }

    # Transform to DataFrames
    try:
        df_primary = bars_to_dataframe(bars_primary)
        df_primary['timeframe'] = '1m'

        # Add default values
        for col in ['bid_volume', 'ask_volume', 'dailyPnL', 'dailyGoal', 'dailyMaxLoss']:
            if col not in df_primary.columns:
                default_values = {
                    'bid_volume': 0.0,
                    'ask_volume': 0.0,
                    'dailyPnL': 0.0,
                    'dailyGoal': request.get('dailyGoal', 500.0),
                    'dailyMaxLoss': request.get('dailyMaxLoss', 250.0)
                }
                df_primary[col] = default_values[col]

        df_secondary = None
        if bars_secondary:
            df_secondary = bars_to_dataframe(bars_secondary)
            df_secondary['timeframe'] = '5m'

            # Add default values to secondary
            for col in ['bid_volume', 'ask_volume', 'dailyPnL', 'dailyGoal', 'dailyMaxLoss']:
                if col not in df_secondary.columns:
                    default_values = {
                        'bid_volume': 0.0,
                        'ask_volume': 0.0,
                        'dailyPnL': 0.0,
                        'dailyGoal': request.get('dailyGoal', 500.0),
                        'dailyMaxLoss': request.get('dailyMaxLoss', 250.0)
                    }
                    df_secondary[col] = default_values[col]

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error parsing data: {e}"
        }

    # Validate DataFrames
    is_valid, error = validate_dataframe(df_primary)
    if not is_valid:
        return {
            "status": "error",
            "message": error
        }

    # Update model with data
    update_model_fn(df_primary, df_secondary)

    # Schedule training
    schedule_training_fn(df_primary)

    return {
        "status": "ok",
        "bars_received": len(df_primary),
        "data_type": request.get('type', 'unknown'),
        "model_training": "scheduled",
        "message": "Training started in background. Check /training-status for progress."
    }


def handle_realtime_request(
    new_bar: pd.DataFrame,
    new_bar_secondary: pd.DataFrame | None,
    request: Dict[str, Any],
    model,
    training_status: Dict[str, Any],
    min_confidence_threshold: float = 0.25
) -> Dict[str, Any]:
    """
    Handle realtime prediction request functionally.

    Args:
        new_bar: New bar data (primary timeframe)
        new_bar_secondary: New bar data (secondary timeframe)
        request: Full request dictionary
        model: Trading model instance (NOT pure, but wrapped)
        training_status: Training status dictionary
        min_confidence_threshold: Minimum confidence for signals

    Returns:
        Response dictionary
    """
    # Update model with new data
    current_data = model.update_historical_data(new_bar, new_bar_secondary)

    print(f"\n{'='*70}")
    print(f"📊 REQUEST HANDLER - CHECKING MODEL STATUS")
    print(f"{'='*70}")
    print(f"Model trained: {model.is_trained}")
    print(f"Training in progress: {training_status['is_training']}")
    print(f"Data bars: {len(current_data)}")
    print(f"{'='*70}\n")

    # Check if predictions should be blocked
    should_block, reason = should_block_prediction_during_training(
        training_status["is_training"],
        model.is_trained
    )

    if should_block:
        print(f"\n🚫 PREDICTION BLOCKED: {reason}")
        print(f"   Model trained: {model.is_trained}")
        print(f"   Training in progress: {training_status['is_training']}\n")
        return {
            "status": "training" if training_status["is_training"] else "not_trained",
            "message": reason,
            "signal": "hold",
            "confidence": 0.0,
            "model_training": training_status["is_training"],
            "model_trained": model.is_trained,
            "training_progress": training_status["progress"]
        }

    # Make prediction
    try:
        print(f"\n{'='*70}")
        print(f"STARTING PREDICTION")
        print(f"{'='*70}")
        print(f"Model trained: {model.is_trained}")
        print(f"Training in progress: {training_status['is_training']}")
        print(f"Data available: {len(current_data)} bars")
        print(f"{'='*70}\n")

        account_balance = request.get('accountBalance', 25000.0)

        # Validate account balance
        is_valid, error = validate_account_balance(account_balance)
        if not is_valid:
            return {
                "status": "error",
                "message": error,
                "signal": "hold",
                "confidence": 0.0
            }

        # Get trade parameters from model FIRST to know the signal direction
        print(f"\n🤖 Getting model prediction to check trend alignment...")
        trade_params = model.predict_with_risk_params(
            current_data,
            account_balance=account_balance
        )

        signal = trade_params['signal']
        confidence = trade_params.get('confidence', 0.0)

        print(f"   Model signal: {signal.upper()}, Confidence: {confidence:.2%}")

        # Check multi-timeframe alignment AFTER getting model signal
        # This prevents counter-trend trades that have negative expected value
        secondary_data = model.historical_data_secondary if hasattr(model, 'historical_data_secondary') else None

        # Only run MTF filter if we have sufficient 5m data
        if secondary_data is not None and len(secondary_data) >= 50:
            # Calculate 5m trend
            close_5m = secondary_data['close'].values
            sma_20_5m = np.mean(close_5m[-20:])
            sma_50_5m = np.mean(close_5m[-50:])

            if sma_20_5m > sma_50_5m * 1.002:  # 0.2% threshold
                trend_5m = "UP"
            elif sma_20_5m < sma_50_5m * 0.998:
                trend_5m = "DOWN"
            else:
                trend_5m = "NEUTRAL"

            # Check alignment between model signal and 5m trend
            mtf_aligned = True
            mtf_reasons = []

            if signal == "long" and trend_5m == "DOWN":
                mtf_aligned = False
                mtf_reasons.append(f"Counter-trend LONG: Model wants LONG but 5m trend is DOWN (SMA20={sma_20_5m:.2f} < SMA50={sma_50_5m:.2f})")
            elif signal == "short" and trend_5m == "UP":
                mtf_aligned = False
                mtf_reasons.append(f"Counter-trend SHORT: Model wants SHORT but 5m trend is UP (SMA20={sma_20_5m:.2f} > SMA50={sma_50_5m:.2f})")
            else:
                mtf_reasons.append(f"Aligned: Model signal={signal.upper()}, 5m trend={trend_5m}")

            if mtf_aligned:
                print(f"\n✅ MTF FILTER: Timeframes aligned - trade allowed")
                print(f"   Model: {signal.upper()}, 5m trend: {trend_5m}")
        else:
            # Not enough 5m data yet - allow trade but log warning
            mtf_aligned = True
            mtf_reasons = [f"Insufficient 5m data ({len(secondary_data) if secondary_data is not None else 0} bars) - MTF filter skipped"]
            print(f"⚠️  MTF FILTER: {mtf_reasons[0]}")

        if not mtf_aligned:
            print(f"\n🚫 MULTI-TIMEFRAME FILTER: Trade blocked")
            for reason in mtf_reasons:
                print(f"   {reason}")
            print(f"   Action: Skipping counter-trend trade\n")

            # Get filter statistics for reporting
            mtf_filter = get_mtf_filter()
            filter_stats = mtf_filter.get_statistics()

            return {
                "status": "ok",
                "signal": "hold",
                "confidence": 0.0,
                "mtf_filtered": True,
                "mtf_reasons": mtf_reasons,
                "mtf_stats": filter_stats,
                "reason": "Multi-timeframe alignment check failed - counter-trend trade rejected",
                "bars_received": 1,
                "data_type": request.get('type', 'unknown')
            }

        # Check market regime before trading
        # IMPORTANT: use_adx=False because we removed ADX indicator in Phase 2
        print(f"\n📊 CHECKING MARKET REGIME...")
        regime = calculate_market_regime(current_data, lookback=20, use_adx=False)
        skip_trading, regime_reason = should_skip_trading(regime)

        print(f"   Regime: {regime['regime'].upper()}")
        print(f"   Should trade: {regime.get('should_trade', False)}")
        print(f"   Skip trading: {skip_trading}")
        if skip_trading:
            print(f"   Reason: {regime_reason}")

        if skip_trading:
            print(f"\n🚫 MARKET REGIME FILTER: {regime_reason}")
            print(f"   Regime: {regime['regime'].upper()}")
            print(f"   Metrics: {regime['metrics']}")
            print(f"   Action: Skipping trade\n")
            return {
                "status": "ok",
                "signal": "hold",
                "confidence": 0.0,
                "regime_filtered": True,
                "regime": regime,
                "reason": regime_reason,
                "bars_received": 1,
                "data_type": request.get('type', 'unknown')
            }

        # Model prediction was already called above for MTF check
        # Use the existing trade_params, signal, and confidence
        print(f"\n✅ Using model prediction from MTF check")

        # DIAGNOSTIC: Log raw prediction before any filtering
        print(f"\n{'='*70}")
        print(f"RAW MODEL PREDICTION")
        print(f"{'='*70}")
        print(f"Signal: {signal.upper()}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"{'='*70}\n")

        # Adjust confidence threshold based on market regime
        adjusted_threshold = get_adjusted_confidence_threshold(
            min_confidence_threshold,
            regime
        )

        if adjusted_threshold != min_confidence_threshold:
            print(f"\n📊 REGIME-ADJUSTED THRESHOLD:")
            print(f"   Regime: {regime['regime'].upper()}")
            print(f"   Base threshold: {min_confidence_threshold:.1%}")
            print(f"   Adjusted threshold: {adjusted_threshold:.1%} (multiplier: {regime['confidence_multiplier']:.2f}x)\n")

        # Apply confidence threshold (use regime-adjusted threshold)
        filtered_signal, was_filtered = apply_confidence_threshold(
            signal,
            confidence,
            adjusted_threshold
        )

        # DIAGNOSTIC: Log confidence filtering
        if was_filtered:
            print(f"\n⚠️  CONFIDENCE FILTER: Signal changed from {signal.upper()} to {filtered_signal.upper()}")
            print(f"   Confidence: {confidence:.4f} < Threshold: {adjusted_threshold:.4f}")
            print(f"   Reason: Insufficient confidence\n")

        # Apply signal stability check to prevent over-trading
        # BUGFIX: Convert signal format from long/short to buy/sell for stability check
        signal_for_stability = {"long": "buy", "short": "sell", "hold": "hold"}.get(filtered_signal, filtered_signal)
        stability_allowed, stability_reason = check_signal_stability(
            signal_for_stability,
            confidence
        )

        if not stability_allowed:
            # Override to hold if signal stability check fails
            filtered_signal = "hold"
            was_filtered = True
            print(f"\n⚠️  SIGNAL STABILITY CHECK: {stability_reason}")
            print(f"   Original signal: {signal.upper()}, Confidence: {confidence:.1%}")
            print(f"   Action: Forcing HOLD to prevent over-trading\n")

        # Get exit analysis
        exit_analysis = {'should_exit': False, 'reason': 'No position', 'urgency': 'none'}
        trailing_stop_price = 0.0
        current_position = request.get('current_position', 'flat')
        entry_price = request.get('entry_price', 0.0)

        if current_position in ['long', 'short'] and entry_price > 0:
            # Calculate trailing stop and exit analysis
            from risk_management import StopTargetCalculator
            stop_calculator = StopTargetCalculator()

            current_price = current_data['close'].iloc[-1] if len(current_data) > 0 else 0.0

            # Get volatility estimate for trailing stops
            # MIGRATED: Replaced ATR (lagging indicator) with pure candle range
            if len(current_data) >= 14:
                # Calculate average true range using pure price action (no smoothing)
                # True range = max(high-low, |high-prev_close|, |low-prev_close|)
                high = current_data['high'].values
                low = current_data['low'].values
                close = current_data['close'].values

                # Calculate true range for last 14 bars
                tr = np.zeros(len(high))
                for i in range(1, len(high)):
                    tr[i] = max(
                        high[i] - low[i],
                        abs(high[i] - close[i-1]),
                        abs(low[i] - close[i-1])
                    )

                # Simple average (no EMA smoothing - pure price action)
                current_atr = np.mean(tr[-14:]) if len(tr) >= 14 else 15.0
            else:
                current_atr = 15.0  # Default fallback

            trailing_stop_price = stop_calculator.calculate_trailing_stop(
                entry_price=entry_price,
                current_price=current_price,
                direction=current_position,
                atr=current_atr,
                initial_stop=request.get('current_stop_loss', 0) or trade_params.get('stop_loss', 0),
                trail_activation_rr=1.0,
                trail_distance_atr=0.75
            )

            exit_analysis = model.detect_early_exit(
                current_data,
                current_position,
                entry_price
            )

        # Determine final signal with exit logic
        final_signal = determine_signal_action(filtered_signal, confidence, exit_analysis)

        # DIAGNOSTIC: Log final signal decision
        print(f"\n{'='*70}")
        print(f"FINAL SIGNAL DECISION")
        print(f"{'='*70}")
        print(f"Raw signal: {signal.upper()}")
        print(f"After confidence filter: {filtered_signal.upper()}")
        print(f"After exit logic: {final_signal.upper()}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"Threshold: {adjusted_threshold:.4f} ({adjusted_threshold*100:.2f}%)")
        print(f"Market regime: {regime['regime'].upper()} (multiplier: {regime['confidence_multiplier']:.2f}x)")
        if was_filtered:
            print(f"⚠️  Signal was filtered due to low confidence")
        print(f"{'='*70}\n")

        # Update signal state for tracking (after final signal is determined)
        # BUGFIX: Convert signal format from long/short to buy/sell for state tracking
        final_signal_for_state = {"long": "buy", "short": "sell", "hold": "hold"}.get(final_signal, final_signal)
        update_signal_state(final_signal_for_state, confidence)

        # Build response using pure functions
        risk_management = format_trade_params_response(trade_params)
        counter_trend_filter = format_counter_trend_filter_response(trade_params, signal, confidence)
        exit_analysis_formatted = format_exit_analysis_response(exit_analysis)
        trailing_stop = format_trailing_stop_response(trailing_stop_price, current_position, entry_price)

        # Create final response
        response = format_prediction_response(
            signal=final_signal,
            confidence=confidence,
            raw_signal=signal,
            filtered=(final_signal != signal),
            risk_management=risk_management,
            exit_analysis=exit_analysis_formatted,
            trailing_stop=trailing_stop,
            counter_trend_filter=counter_trend_filter
        )

        response["status"] = "ok"
        response["bars_received"] = 1
        response["data_type"] = request.get('type', 'unknown')
        response["confidence_threshold"] = adjusted_threshold
        response["base_confidence_threshold"] = min_confidence_threshold

        # Add regime information to response
        response["market_regime"] = {
            "regime": regime["regime"],
            "should_trade": regime["should_trade"],
            "confidence_multiplier": regime["confidence_multiplier"],
            "description": get_regime_description(regime),
            "metrics": regime["metrics"]
        }

        return response

    except Exception as e:
        return {
            "status": "error",
            "message": f"Prediction failed: {e}",
            "signal": "hold",
            "confidence": 0.0
        }


def create_response_builder(min_confidence_threshold: float = 0.25):
    """
    Create a response builder function with preset configuration.

    This demonstrates functional composition and partial application.

    Args:
        min_confidence_threshold: Minimum confidence threshold

    Returns:
        Function that builds responses with preset configuration
    """
    def build_response(
        new_bar: pd.DataFrame,
        new_bar_secondary: pd.DataFrame | None,
        request: Dict[str, Any],
        model,
        training_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        return handle_realtime_request(
            new_bar,
            new_bar_secondary,
            request,
            model,
            training_status,
            min_confidence_threshold
        )

    return build_response
