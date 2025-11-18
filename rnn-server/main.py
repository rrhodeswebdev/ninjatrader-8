"""
FastAPI server for RNN-based trading predictions.

This server provides endpoints for:
- Historical data ingestion and model training
- Real-time trading signal predictions
- Model health checks and status monitoring
"""

import logging
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime
from model import TradingModel
import asyncio
import math

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RNN Trading Server", version="2.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("="*70)
logger.info("🚀 INITIALIZING RNN TRADING SERVER")
logger.info("="*70)

# Initialize trading model
trading_model = TradingModel(sequence_length=15)

# Confidence threshold for signal filtering
MIN_CONFIDENCE_THRESHOLD = 0.55
logger.info(f"🎯 Minimum confidence threshold: {MIN_CONFIDENCE_THRESHOLD}")

# Track training status
training_status = {
    "is_training": False,
    "progress": "",
    "error": None
}

logger.info("="*70)
logger.info("✅ SERVER INITIALIZATION COMPLETE")
logger.info("="*70)

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

@app.get("/")
def read_root():
    """Root endpoint - server info."""
    return {
        "service": "RNN Trading Server",
        "version": "2.0",
        "status": "running",
        "model_trained": trading_model.is_trained
    }

def train_model_background(df: pd.DataFrame):
    """Background task for model training."""
    global training_status

    try:
        training_status["is_training"] = True
        training_status["progress"] = "Training started..."
        training_status["error"] = None

        logger.info("="*70)
        logger.info("🎓 BACKGROUND TRAINING STARTED")
        logger.info("="*70)
        logger.info(f"📊 Training data: {len(df)} bars")

        trading_model.train(df, epochs=100, batch_size=32)

        logger.info("="*70)
        logger.info("💾 SAVING TRAINED MODEL")
        logger.info("="*70)
        trading_model.save_model()

        training_status["is_training"] = False
        training_status["progress"] = "Training complete - model saved"

        logger.info("="*70)
        logger.info("✅ BACKGROUND TRAINING COMPLETE")
        logger.info(f"   Model trained: {trading_model.is_trained}")
        logger.info(f"   Model saved: models/trading_model.pth")
        logger.info("   Ready for predictions!")
        logger.info("="*70)

    except Exception as e:
        training_status["is_training"] = False
        training_status["error"] = str(e)
        training_status["progress"] = f"Training failed: {e}"
        logger.error(f"❌ Training error: {e}", exc_info=True)


@app.post("/analysis")
async def analysis(request: dict, background_tasks: BackgroundTasks):
    """
    Main analysis endpoint for historical data and real-time predictions.

    Handles two request types:
    - Historical: Ingests data and schedules model training
    - Realtime: Generates trading signals from current market data
    """
    logger.info("="*70)
    logger.info("📥 RECEIVED REQUEST AT /analysis")
    logger.info("="*70)

    # Determine request type
    request_type = extract_request_type(request)

    if request_type == "historical":
        # Extract and process historical data
        bars_primary, bars_secondary = extract_bars_from_request(request)

        logger.info(f"📊 Request Type: HISTORICAL")
        logger.info(f"   Primary bars: {len(bars_primary)}")
        logger.info(f"   Secondary bars: {len(bars_secondary)}")

        # Handle historical data ingestion
        response = handle_historical_request(
            bars_primary=bars_primary,
            bars_secondary=bars_secondary,
            request=request,
            update_model_fn=trading_model.update_historical_data,
            schedule_training_fn=lambda df: background_tasks.add_task(train_model_background, df)
        )

        logger.info(f"✅ Response Status: {response.get('status')}")
        logger.info("="*70)

        return response

    else:
        # Realtime prediction request
        logger.info(f"🔮 Request Type: REALTIME PREDICTION")

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

        logger.debug(f"📊 New Bar Data:\n{new_bar.to_string(index=False)}")
        if new_bar_secondary is not None:
            logger.debug(f"📊 Secondary Bar Data:\n{new_bar_secondary.to_string(index=False)}")

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
            logger.info("="*70)
            logger.info("🎯 PREDICTION RESULT")
            logger.info("="*70)
            logger.info(f"📊 Signal: {response['signal'].upper()}")
            logger.info(f"📈 Confidence: {response['confidence']:.4f} ({response['confidence']*100:.2f}%)")

            if response.get('filtered'):
                logger.info(f"🔄 Filtered from: {response['raw_signal'].upper()}")

            risk = response.get('risk_management', {})
            if risk.get('contracts', 0) > 0:
                logger.info("💰 RISK MANAGEMENT:")
                logger.info(f"   Contracts: {risk['contracts']}")
                logger.info(f"   Entry: ${risk['entry_price']:.2f}")
                logger.info(f"   Stop Loss: ${risk['stop_loss']:.2f}")
                logger.info(f"   Take Profit: ${risk['take_profit']:.2f}")
                logger.info(f"   Risk/Reward: {risk.get('risk_reward_ratio', 0):.2f}")

            logger.info("="*70)

        # Sanitize all floats before returning
        try:
            sanitized = sanitize_dict_floats(response)
            return sanitized
        except Exception as e:
            logger.error(f"❌ ERROR SANITIZING RESPONSE: {e}", exc_info=True)
            return {
                "status": "error",
                "signal": "hold",
                "confidence": 0.0,
                "message": f"Response serialization error: {e}"
            }


@app.get("/health-check")
def health_check():
    """Health check endpoint - returns server and model status."""
    logger.info(f"🏥 Health check - Model trained: {trading_model.is_trained}")
    return {
        "status": "ok",
        "model_trained": trading_model.is_trained,
        "device": str(trading_model.device),
        "version": "2.0",
        "sequence_length": trading_model.sequence_length,
        "min_confidence": MIN_CONFIDENCE_THRESHOLD
    }


@app.get("/training-status")
def get_training_status():
    """Get current training status and progress."""
    return training_status


@app.post("/save-model")
def save_model():
    """Manually save the trained model to disk."""
    try:
        logger.info("💾 Saving model...")
        trading_model.save_model()
        logger.info("✅ Model saved successfully")
        return {"status": "ok", "message": "Model saved successfully"}
    except Exception as e:
        logger.error(f"❌ Error saving model: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.post("/load-model")
def load_model():
    """Manually load a trained model from disk."""
    try:
        logger.info("📂 Loading model...")
        success = trading_model.load_model()
        if success:
            logger.info("✅ Model loaded successfully")
            return {"status": "ok", "message": "Model loaded successfully"}
        else:
            logger.warning("⚠️  Model file not found")
            return {"status": "error", "message": "Model not found"}
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
