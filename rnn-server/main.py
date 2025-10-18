from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime
from model import TradingModel
import asyncio
import math

app = FastAPI()

def sanitize_float(value):
    """Convert inf/nan to valid JSON numbers"""
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)

# Initialize model instance with IMPROVED settings
# IMPROVED: Sequence length increased from 12 to 15 for better pattern recognition
# (Previous: 20 was too long, 12 was too short - 15 is the sweet spot)
trading_model = TradingModel(sequence_length=15)

# IMPROVED: Confidence threshold for predictions
# LOWERED from 0.40 back to 0.25 to allow more trading opportunities
# NOTE: 0.40 was filtering out too many valid signals
# The Round 2 improvements have better confidence calibration, so 0.25 is safe
MIN_CONFIDENCE_THRESHOLD = 0.25  # 25% minimum (was 0.40, reduced for more signals)

# Track training status
training_status = {
    "is_training": False,
    "progress": "",
    "error": None
}

# Define request models
class BarData(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = 0.0
    bid_volume: Optional[float] = 0.0
    ask_volume: Optional[float] = 0.0

class HistoricalRequest(BaseModel):
    bars: Optional[List[BarData]] = None  # Legacy support
    primary_bars: Optional[List[BarData]] = None  # Multi-timeframe
    secondary_bars: Optional[List[BarData]] = None  # Multi-timeframe
    type: str

class RealtimeRequest(BaseModel):
    # Legacy single timeframe support
    time: Optional[str] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = 0.0

    # Multi-timeframe support
    primary_bar: Optional[dict] = None
    secondary_bar: Optional[dict] = None

    type: str
    bid_volume: Optional[float] = 0.0
    ask_volume: Optional[float] = 0.0
    dailyPnL: Optional[float] = 0.0
    dailyGoal: Optional[float] = 0.0
    dailyMaxLoss: Optional[float] = 0.0

    # Position information for exit analysis
    current_position: Optional[str] = "flat"  # "flat", "long", or "short"
    entry_price: Optional[float] = 0.0
    position_quantity: Optional[int] = 0

@app.get("/")
def read_root():
    return {"Hello": "World"}

def train_model_background(df: pd.DataFrame):
    """Background task for training the model"""
    global training_status

    try:
        training_status["is_training"] = True
        training_status["progress"] = "Training started..."
        training_status["error"] = None

        print("\n" + "="*70)
        print("BACKGROUND TRAINING STARTED")
        print("="*70)

        # Train the model
        trading_model.train(df, epochs=100, batch_size=32)

        # Explicitly save the model after training
        print("\n" + "="*70)
        print("SAVING TRAINED MODEL")
        print("="*70)
        trading_model.save_model()

        # Verify model is marked as trained
        print(f"Model is_trained status: {trading_model.is_trained}")

        training_status["is_training"] = False
        training_status["progress"] = "Training complete - model saved"

        print("\n" + "="*70)
        print("✅ BACKGROUND TRAINING COMPLETE")
        print(f"   Model trained: {trading_model.is_trained}")
        print(f"   Model saved: models/trading_model.pth")
        print("   Ready for predictions!")
        print("="*70 + "\n")

    except Exception as e:
        training_status["is_training"] = False
        training_status["error"] = str(e)
        training_status["progress"] = f"Training failed: {e}"
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()


@app.post("/analysis")
async def analysis(request: dict, background_tasks: BackgroundTasks):
    """
    Accepts both historical (bulk) and realtime (single bar) data
    and converts to pandas DataFrame for RNN processing
    """
    print("\n" + "="*50)
    print("RECEIVED DATA AT /analysis")
    print("="*50)

    # Check if this is historical or realtime data
    if "bars" in request or "primary_bars" in request:
        # Historical data - multiple bars
        bars_list = request.get('primary_bars') or request.get('bars')  # Support both formats
        secondary_bars_list = request.get('secondary_bars', [])

        print(f"Data Type: HISTORICAL ({len(bars_list)} primary bars, {len(secondary_bars_list)} secondary bars)")

        # Validate input
        if len(bars_list) == 0:
            return {
                "status": "error",
                "message": "No bars received"
            }

        # Convert primary timeframe to DataFrame
        try:
            df_primary = pd.DataFrame([
                {
                    'time': bar['time'],
                    'open': bar['open'],
                    'high': bar['high'],
                    'low': bar['low'],
                    'close': bar['close'],
                    'volume': bar.get('volume', 0.0),
                    'bid_volume': bar.get('bid_volume', 0.0),
                    'ask_volume': bar.get('ask_volume', 0.0),
                    'dailyPnL': 0.0,
                    'dailyGoal': request.get('dailyGoal', 500.0),
                    'dailyMaxLoss': request.get('dailyMaxLoss', 250.0)
                }
                for bar in bars_list
            ])
            df_primary['time'] = pd.to_datetime(df_primary['time'])
            df_primary['timeframe'] = '1m'  # Mark as primary timeframe

            # Convert secondary timeframe to DataFrame if available
            df_secondary = None
            if len(secondary_bars_list) > 0:
                df_secondary = pd.DataFrame([
                    {
                        'time': bar['time'],
                        'open': bar['open'],
                        'high': bar['high'],
                        'low': bar['low'],
                        'close': bar['close'],
                        'volume': bar.get('volume', 0.0),
                        'bid_volume': bar.get('bid_volume', 0.0),
                        'ask_volume': bar.get('ask_volume', 0.0),
                        'dailyPnL': 0.0,
                        'dailyGoal': request.get('dailyGoal', 500.0),
                        'dailyMaxLoss': request.get('dailyMaxLoss', 250.0)
                    }
                    for bar in secondary_bars_list
                ])
                df_secondary['time'] = pd.to_datetime(df_secondary['time'])
                df_secondary['timeframe'] = '5m'  # Mark as secondary timeframe

            # Use primary dataframe as main df for now (model will use both)
            df = df_primary

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error parsing data: {e}"
            }

        print("\nPrimary DataFrame Info:")
        print(df.info())
        print("\nPrimary DataFrame Head:")
        print(df.head(10))
        print("\nPrimary DataFrame Tail:")
        print(df.tail(10))
        print("\nPrimary DataFrame Stats:")
        print(df.describe())

        # Log secondary dataframe if available
        if df_secondary is not None:
            print("\n" + "="*50)
            print("SECONDARY TIMEFRAME (5-min)")
            print("="*50)
            print("\nSecondary DataFrame Info:")
            print(df_secondary.info())
            print("\nSecondary DataFrame Head:")
            print(df_secondary.head(10))
            print("\nSecondary DataFrame Tail:")
            print(df_secondary.tail(10))
            print("="*50)

        # Update historical data in model (both timeframes)
        trading_model.update_historical_data(df, df_secondary)

        # Train the model in background (non-blocking)
        background_tasks.add_task(train_model_background, df)

        print("="*50)
        print(f"DataFrame Shape: {df.shape}")
        print("Training scheduled in background...")
        print("="*50 + "\n")

        return {
            "status": "ok",
            "bars_received": len(df),
            "data_type": request.get('type', 'unknown'),
            "model_training": "scheduled",
            "message": "Training started in background. Check /training-status for progress."
        }

    else:
        # Realtime data - single bar (or multi-timeframe bars)
        print(f"Data Type: REALTIME")

        # Check for multi-timeframe format
        if 'primary_bar' in request and request['primary_bar']:
            # Multi-timeframe format
            primary_bar_data = request['primary_bar']
            secondary_bar_data = request.get('secondary_bar', {})

            new_bar_primary = pd.DataFrame([{
                'time': primary_bar_data['time'],
                'open': primary_bar_data['open'],
                'high': primary_bar_data['high'],
                'low': primary_bar_data['low'],
                'close': primary_bar_data['close'],
                'volume': primary_bar_data.get('volume', 0.0),
                'bid_volume': request.get('bid_volume', 0.0),
                'ask_volume': request.get('ask_volume', 0.0),
                'dailyPnL': request.get('dailyPnL', 0.0),
                'dailyGoal': request.get('dailyGoal', 0.0),
                'dailyMaxLoss': request.get('dailyMaxLoss', 0.0),
                'timeframe': '1m'
            }])
            new_bar_primary['time'] = pd.to_datetime(new_bar_primary['time'])

            new_bar_secondary = None
            if secondary_bar_data:
                new_bar_secondary = pd.DataFrame([{
                    'time': secondary_bar_data['time'],
                    'open': secondary_bar_data['open'],
                    'high': secondary_bar_data['high'],
                    'low': secondary_bar_data['low'],
                    'close': secondary_bar_data['close'],
                    'volume': secondary_bar_data.get('volume', 0.0),
                    'bid_volume': 0.0,
                    'ask_volume': 0.0,
                    'dailyPnL': request.get('dailyPnL', 0.0),
                    'dailyGoal': request.get('dailyGoal', 0.0),
                    'dailyMaxLoss': request.get('dailyMaxLoss', 0.0),
                    'timeframe': '5m'
                }])
                new_bar_secondary['time'] = pd.to_datetime(new_bar_secondary['time'])

            new_bar = new_bar_primary
        else:
            # Legacy single timeframe format
            required_fields = ['time', 'open', 'high', 'low', 'close']
            if not all(field in request for field in required_fields):
                return {
                    "status": "error",
                    "message": f"Missing required fields. Need: {required_fields}"
                }

            new_bar = pd.DataFrame([{
                'time': request['time'],
                'open': request['open'],
                'high': request['high'],
                'low': request['low'],
                'close': request['close'],
                'volume': request.get('volume', 0.0),
                'bid_volume': request.get('bid_volume', 0.0),
                'ask_volume': request.get('ask_volume', 0.0),
                'dailyPnL': request.get('dailyPnL', 0.0),
                'dailyGoal': request.get('dailyGoal', 0.0),
                'dailyMaxLoss': request.get('dailyMaxLoss', 0.0),
                'timeframe': '1m'
            }])
            new_bar['time'] = pd.to_datetime(new_bar['time'])
            new_bar_secondary = None

        print("\nNew Bar Data:")
        print(new_bar.to_string(index=False))
        if new_bar_secondary is not None:
            print("\nSecondary Bar Data:")
            print(new_bar_secondary.to_string(index=False))

        # Update historical data (both timeframes)
        current_data = trading_model.update_historical_data(new_bar, new_bar_secondary)

        # CHECK IF MODEL IS TRAINING - Block predictions during training
        if training_status["is_training"]:
            print("\n" + "="*50)
            print("⚠️  MODEL IS TRAINING - Returning HOLD signal")
            print("="*50 + "\n")
            return {
                "status": "training",
                "message": "Model is currently training. No predictions available.",
                "signal": "hold",
                "confidence": 0.0,
                "model_training": True,
                "training_progress": training_status["progress"]
            }

        # CHECK IF MODEL IS TRAINED - Block predictions if model not ready
        if not trading_model.is_trained:
            print("\n" + "="*50)
            print("⚠️  MODEL NOT TRAINED - Returning HOLD signal")
            print("="*50)
            print(f"   Model is_trained flag: {trading_model.is_trained}")
            print(f"   Training status: {training_status['is_training']}")
            print(f"   Training progress: {training_status['progress']}")

            # Check if model file exists
            import os
            model_exists = os.path.exists('models/trading_model.pth')
            print(f"   Model file exists: {model_exists}")

            if model_exists:
                print("   ⚠️  Model file exists but is_trained=False!")
                print("   Try reloading the model or check training logs")

            print("="*50 + "\n")
            return {
                "status": "not_trained",
                "message": "Model is not trained yet. No predictions available.",
                "signal": "hold",
                "confidence": 0.0,
                "model_trained": False,
                "training_status": training_status["progress"]
            }

        # CHECK DAILY MAX LOSS - Block new trades if daily loss limit hit
        daily_pnl = request.get('dailyPnL', 0.0)
        daily_max_loss = request.get('dailyMaxLoss', 0.0)

        if daily_max_loss > 0 and daily_pnl <= -daily_max_loss:
            print("\n" + "="*50)
            print("🛑 DAILY MAX LOSS HIT - BLOCKING NEW TRADES")
            print("="*50)
            print(f"Daily P&L: ${daily_pnl:.2f}")
            print(f"Max Loss Limit: ${daily_max_loss:.2f}")
            print(f"Loss Exceeded By: ${abs(daily_pnl + daily_max_loss):.2f}")
            print("Returning HOLD to prevent new entries")
            print("="*50 + "\n")

            return {
                "status": "max_loss_hit",
                "message": f"Daily max loss of ${daily_max_loss:.2f} has been exceeded. Current P&L: ${daily_pnl:.2f}",
                "signal": "hold",
                "confidence": 0.0,
                "daily_pnl": sanitize_float(daily_pnl),
                "daily_max_loss": sanitize_float(daily_max_loss),
                "loss_exceeded": True,
                "exceeded_by": sanitize_float(abs(daily_pnl + daily_max_loss))
            }

        # Make prediction with risk management parameters
        try:
            # Get account balance from request (default to $25,000)
            account_balance = request.get('accountBalance', 25000.0)

            # Get full trade parameters with risk management
            trade_params = trading_model.predict_with_risk_params(
                current_data,
                account_balance=account_balance
            )

            signal = trade_params['signal']
            confidence = trade_params['confidence']

            # Apply confidence threshold filtering (for backward compatibility)
            filtered_signal = signal
            if confidence < MIN_CONFIDENCE_THRESHOLD:
                filtered_signal = "hold"
                print(f"\n⚠️  Low confidence ({confidence*100:.2f}%) - Filtering {signal.upper()} → HOLD")

            # Check for exit conditions if in a position
            exit_analysis = {'should_exit': False, 'reason': 'No position', 'urgency': 'none'}
            current_position = request.get('current_position', 'flat')
            entry_price = request.get('entry_price', 0.0)

            if current_position in ['long', 'short'] and entry_price > 0:
                exit_analysis = trading_model.detect_early_exit(
                    current_data,
                    current_position,
                    entry_price
                )

                # If exit detected, override signal to HOLD (to trigger exit)
                if exit_analysis['should_exit']:
                    filtered_signal = "hold"
                    print(f"\n🚨 EARLY EXIT DETECTED 🚨")
                    print(f"Reason: {exit_analysis['reason']}")
                    print(f"Urgency: {exit_analysis['urgency'].upper()}")

            print("\n" + "="*50)
            print("PREDICTION WITH RISK PARAMETERS")
            print("="*50)
            print(f"Raw Signal: {signal.upper()}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"Final Signal: {filtered_signal.upper()}")
            if filtered_signal != signal:
                if exit_analysis['should_exit']:
                    print(f"(Overridden to HOLD due to exit condition)")
                else:
                    print(f"(Filtered due to low confidence < {MIN_CONFIDENCE_THRESHOLD*100:.0f}%)")

            # Log risk management parameters
            if trade_params['contracts'] > 0:
                print(f"\n📊 RISK MANAGEMENT PARAMETERS:")
                print(f"  Contracts: {trade_params['contracts']}")
                print(f"  Entry Price: ${trade_params['entry_price']:.2f}")
                print(f"  Stop Loss: ${trade_params['stop_loss']:.2f}")
                print(f"  Take Profit: ${trade_params['take_profit']:.2f}")
                print(f"  Risk/Reward: {trade_params.get('risk_reward', 0):.2f}")
                print(f"  Dollar Risk: ${trade_params.get('risk_dollars', 0):.2f}")
                print(f"  Risk %: {trade_params.get('risk_pct', 0)*100:.2f}%")

            print("="*50 + "\n")

            return {
                "status": "ok",
                "bars_received": 1,
                "data_type": request.get('type', 'unknown'),
                "signal": filtered_signal,
                "raw_signal": signal,
                "confidence": sanitize_float(confidence),
                "confidence_threshold": MIN_CONFIDENCE_THRESHOLD,
                "filtered": filtered_signal != signal,
                "recommendation": f"{filtered_signal.upper()} with {sanitize_float(confidence)*100:.1f}% confidence" +
                                 (f" (filtered from {signal.upper()})" if filtered_signal != signal else ""),
                # Risk management parameters
                "risk_management": {
                    "contracts": trade_params['contracts'],
                    "entry_price": sanitize_float(trade_params['entry_price']),
                    "stop_loss": sanitize_float(trade_params['stop_loss']),
                    "take_profit": sanitize_float(trade_params['take_profit']),
                    "stop_distance": sanitize_float(trade_params.get('stop_distance', 0)),
                    "target_distance": sanitize_float(trade_params.get('target_distance', 0)),
                    "risk_reward_ratio": sanitize_float(trade_params.get('risk_reward', 0)),
                    "risk_dollars": sanitize_float(trade_params.get('risk_dollars', 0)),
                    "risk_pct": sanitize_float(trade_params.get('risk_pct', 0)),
                    "regime": trade_params.get('regime', 'unknown')
                },
                # Exit analysis (NEW!)
                "exit_analysis": {
                    "should_exit": exit_analysis['should_exit'],
                    "reason": exit_analysis['reason'],
                    "urgency": exit_analysis['urgency']
                }
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Prediction failed: {e}",
                "signal": "hold",
                "confidence": 0.0
            }

@app.get("/health-check")
def health_check():
    return {
        "status": "ok",
        "model_trained": trading_model.is_trained,
        "device": str(trading_model.device)
    }


@app.get("/training-status")
def get_training_status():
    """Get current training status"""
    return training_status


@app.post("/save-model")
def save_model():
    """Manually save the model"""
    try:
        trading_model.save_model()
        return {"status": "ok", "message": "Model saved successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/load-model")
def load_model():
    """Manually load the model"""
    try:
        success = trading_model.load_model()
        if success:
            return {"status": "ok", "message": "Model loaded successfully"}
        else:
            return {"status": "error", "message": "Model not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
