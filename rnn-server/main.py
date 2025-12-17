"""
FastAPI server for RNN-based trading predictions - FUNCTIONAL PROGRAMMING VERSION

This is a refactored version of main.py using functional programming patterns.
The original main.py is preserved as main_backup.py for reference.
"""

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime
from model import TradingModel
import asyncio
import math

# Import configuration
try:
    from config import CONTRACT as DEFAULT_CONTRACT
except ImportError:
    DEFAULT_CONTRACT = 'MNQ'  # Fallback if config.py doesn't exist

# Import contract specifications
try:
    from contract_specs import get_contract
    CONTRACTS_AVAILABLE = True
except ImportError:
    CONTRACTS_AVAILABLE = False

# Import functional modules
from core.validation import (
    validate_bar_data,
    validate_bars_list,
    validate_dataframe,
)
from core.transformations import (
    sanitize_float,
    sanitize_dict_floats,
    bars_to_dataframe,
    extract_request_type,
    extract_bars_from_request,
    bar_to_dict,
)
from core.prediction import (
    apply_confidence_threshold,
    should_block_prediction_during_training,
)
from services.request_handler import (
    handle_historical_request,
    handle_realtime_request,
)

app = FastAPI()

# Add CORS middleware to allow requests from local tools or future UIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model instance with IMPROVED settings
trading_model = TradingModel(sequence_length=15)

# IMPROVED: Confidence threshold for predictions
# INCREASED: Set to 0.55 for better quality signals and fewer false entries
# Higher threshold = fewer but higher quality trades
# Was 0.25 (testing), now 0.55 (production quality)
MIN_CONFIDENCE_THRESHOLD = 0.8

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
    bars: Optional[List[BarData]] = None
    primary_bars: Optional[List[BarData]] = None
    secondary_bars: Optional[List[BarData]] = None
    type: str

class RealtimeRequest(BaseModel):
    time: Optional[str] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = 0.0
    primary_bar: Optional[dict] = None
    secondary_bar: Optional[dict] = None
    type: str
    bid_volume: Optional[float] = 0.0
    ask_volume: Optional[float] = 0.0
    dailyPnL: Optional[float] = 0.0
    dailyGoal: Optional[float] = 0.0
    dailyMaxLoss: Optional[float] = 0.0
    current_position: Optional[str] = "flat"
    entry_price: Optional[float] = 0.0
    position_quantity: Optional[int] = 0
    contract: Optional[str] = None  # NEW: Contract symbol (ES, NQ, MNQ, etc.)

@app.get("/")
def read_root():
    return {"Hello": "World", "version": "functional"}

def train_model_background(df: pd.DataFrame):
    """Background task for training the model (kept as-is for compatibility)"""
    global training_status

    try:
        training_status["is_training"] = True
        training_status["progress"] = "Training started..."
        training_status["error"] = None

        print("\n" + "="*70)
        print("BACKGROUND TRAINING STARTED")
        print("="*70)

        trading_model.train(df, epochs=100, batch_size=32)

        print("\n" + "="*70)
        print("SAVING TRAINED MODEL")
        print("="*70)
        trading_model.save_model()

        print(f"Model is_trained status: {trading_model.is_trained}")

        training_status["is_training"] = False
        training_status["progress"] = "Training complete - model saved"

        print("\n" + "="*70)
        print(" BACKGROUND TRAINING COMPLETE")
        print(f"   Model trained: {trading_model.is_trained}")
        print(f"   Model saved: models/trading_model.pth")
        print("   Ready for predictions!")
        print("="*70 + "\n")

    except Exception as e:
        training_status["is_training"] = False
        training_status["error"] = str(e)
        training_status["progress"] = f"Training failed: {e}"
        print(f"\n Training error: {e}")
        import traceback
        traceback.print_exc()


@app.post("/analysis")
async def analysis(request: dict, background_tasks: BackgroundTasks):
    """
    Refactored analysis endpoint using functional programming patterns.

    Uses pure functions from core/ and services/ modules for:
    - Validation
    - Data transformation
    - Request handling
    """
    print("\n" + "="*50)
    print("RECEIVED DATA AT /analysis (FUNCTIONAL VERSION)")
    print("="*50)

    # Determine request type using pure function
    request_type = extract_request_type(request)

    if request_type == "historical":
        # Extract bars using pure function
        bars_primary, bars_secondary = extract_bars_from_request(request)

        print(f"Data Type: HISTORICAL ({len(bars_primary)} primary bars, {len(bars_secondary)} secondary bars)")

        # Use functional request handler
        response = handle_historical_request(
            bars_primary=bars_primary,
            bars_secondary=bars_secondary,
            request=request,
            update_model_fn=trading_model.update_historical_data,
            schedule_training_fn=lambda df: background_tasks.add_task(train_model_background, df)
        )

        print("="*50)
        print(f"Response Status: {response.get('status')}")
        print("="*50 + "\n")

        return response

    else:
        # Realtime data handling
        print(f"\n{'='*70}")
        print(f" RECEIVED REALTIME REQUEST")
        print(f"{'='*70}")
        print(f"Data Type: REALTIME")

        # Build bar DataFrames
        if 'primary_bar' in request and request['primary_bar']:
            # Multi-timeframe format
            primary_bar_data = request['primary_bar']
            secondary_bar_data = request.get('secondary_bar', {})

            new_bar_primary = pd.DataFrame([bar_to_dict(
                primary_bar_data,
                bid_volume=request.get('bid_volume', 0.0),
                ask_volume=request.get('ask_volume', 0.0),
                daily_pnl=request.get('dailyPnL', 0.0),
                daily_goal=request.get('dailyGoal', 0.0),
                daily_max_loss=request.get('dailyMaxLoss', 0.0),
                timeframe='1m'
            )])
            new_bar_primary['time'] = pd.to_datetime(new_bar_primary['time'])

            new_bar_secondary = None
            if secondary_bar_data:
                new_bar_secondary = pd.DataFrame([bar_to_dict(
                    secondary_bar_data,
                    daily_pnl=request.get('dailyPnL', 0.0),
                    daily_goal=request.get('dailyGoal', 0.0),
                    daily_max_loss=request.get('dailyMaxLoss', 0.0),
                    timeframe='5m'
                )])
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

            new_bar = pd.DataFrame([bar_to_dict(
                request,
                bid_volume=request.get('bid_volume', 0.0),
                ask_volume=request.get('ask_volume', 0.0),
                daily_pnl=request.get('dailyPnL', 0.0),
                daily_goal=request.get('dailyGoal', 0.0),
                daily_max_loss=request.get('dailyMaxLoss', 0.0),
                timeframe='1m'
            )])
            new_bar['time'] = pd.to_datetime(new_bar['time'])
            new_bar_secondary = None

        print("\nNew Bar Data:")
        print(new_bar.to_string(index=False))
        if new_bar_secondary is not None:
            print("\nSecondary Bar Data:")
            print(new_bar_secondary.to_string(index=False))

        # Use functional request handler for prediction
        response = handle_realtime_request(
            new_bar=new_bar,
            new_bar_secondary=new_bar_secondary,
            request=request,
            model=trading_model,
            training_status=training_status,
            min_confidence_threshold=MIN_CONFIDENCE_THRESHOLD
        )

        # Log prediction results
        if response.get('status') == 'ok':
            print("\n" + "="*50)
            print("PREDICTION WITH RISK PARAMETERS")
            print("="*50)
            print(f"Signal: {response['signal'].upper()}")
            print(f"Confidence: {response['confidence']:.4f} ({response['confidence']*100:.2f}%)")

            if response.get('filtered'):
                print(f"(Filtered from {response['raw_signal'].upper()})")

            risk = response.get('risk_management', {})
            if risk.get('contracts', 0) > 0:
                print(f"\n RISK MANAGEMENT PARAMETERS:")
                print(f"  Contracts: {risk['contracts']}")
                print(f"  Entry Price: ${risk['entry_price']:.2f}")
                print(f"  Stop Loss: ${risk['stop_loss']:.2f}")
                print(f"  Take Profit: ${risk['take_profit']:.2f}")
                print(f"  Risk/Reward: {risk.get('risk_reward_ratio', 0):.2f}")

            print("="*50 + "\n")

        # Sanitize all floats before returning
        try:
            sanitized = sanitize_dict_floats(response)
            return sanitized
        except Exception as e:
            print(f"\n ERROR SANITIZING RESPONSE: {e}")
            import traceback
            traceback.print_exc()
            # Return a basic error response
            return {
                "status": "error",
                "signal": "hold",
                "confidence": 0.0,
                "message": f"Response serialization error: {e}"
            }


@app.get("/health-check")
def health_check():
    """Health check endpoint"""
    print(f"\n[HEALTH CHECK] Model is_trained: {trading_model.is_trained}")
    return {
        "status": "ok",
        "model_trained": trading_model.is_trained,
        "device": str(trading_model.device),
        "version": "functional"
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
