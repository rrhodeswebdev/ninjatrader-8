from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime
from model import TradingModel
import asyncio

app = FastAPI()

# Initialize model instance (replaces global state)
trading_model = TradingModel(sequence_length=20)

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

class HistoricalRequest(BaseModel):
    bars: List[BarData]
    type: str

class RealtimeRequest(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    type: str

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

        # Train the model
        trading_model.train(df, epochs=100, batch_size=32)

        training_status["is_training"] = False
        training_status["progress"] = "Training complete"

    except Exception as e:
        training_status["is_training"] = False
        training_status["error"] = str(e)
        training_status["progress"] = f"Training failed: {e}"
        print(f"Training error: {e}")


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
    if "bars" in request:
        # Historical data - multiple bars
        print(f"Data Type: HISTORICAL ({len(request['bars'])} bars)")

        # Validate input
        if len(request['bars']) == 0:
            return {
                "status": "error",
                "message": "No bars received"
            }

        # Convert to DataFrame
        try:
            df = pd.DataFrame([
                {
                    'time': bar['time'],
                    'open': bar['open'],
                    'high': bar['high'],
                    'low': bar['low'],
                    'close': bar['close']
                }
                for bar in request['bars']
            ])

            # Convert time to datetime
            df['time'] = pd.to_datetime(df['time'])

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error parsing data: {e}"
            }

        print("\nDataFrame Info:")
        print(df.info())
        print("\nDataFrame Head:")
        print(df.head(10))
        print("\nDataFrame Tail:")
        print(df.tail(10))
        print("\nDataFrame Stats:")
        print(df.describe())

        # Update historical data in model
        trading_model.update_historical_data(df)

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
        # Realtime data - single bar
        print(f"Data Type: REALTIME (single bar)")

        # Validate input
        required_fields = ['time', 'open', 'high', 'low', 'close']
        if not all(field in request for field in required_fields):
            return {
                "status": "error",
                "message": f"Missing required fields. Need: {required_fields}"
            }

        # Convert to DataFrame
        try:
            new_bar = pd.DataFrame([{
                'time': request['time'],
                'open': request['open'],
                'high': request['high'],
                'low': request['low'],
                'close': request['close']
            }])

            # Convert time to datetime
            new_bar['time'] = pd.to_datetime(new_bar['time'])

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error parsing data: {e}"
            }

        print("\nNew Bar Data:")
        print(new_bar.to_string(index=False))

        # Update historical data
        current_data = trading_model.update_historical_data(new_bar)

        # Make prediction
        try:
            signal, confidence = trading_model.predict(current_data)

            print("\n" + "="*50)
            print("PREDICTION")
            print("="*50)
            print(f"Signal: {signal.upper()}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print("="*50 + "\n")

            return {
                "status": "ok",
                "bars_received": 1,
                "data_type": request.get('type', 'unknown'),
                "signal": signal,
                "confidence": confidence,
                "recommendation": f"{signal.upper()} with {confidence*100:.1f}% confidence"
            }

        except Exception as e:
            print(f"Prediction error: {e}")
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
