from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime
from model import trading_model

app = FastAPI()

# Store historical data globally for building context
historical_data = None

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

@app.post("/analysis")
def analysis(request: dict):
    """
    Accepts both historical (bulk) and realtime (single bar) data
    and converts to pandas DataFrame for RNN processing
    """
    global historical_data

    print("\n" + "="*50)
    print("RECEIVED DATA AT /analysis")
    print("="*50)

    # Check if this is historical or realtime data
    if "bars" in request:
        # Historical data - multiple bars
        print(f"Data Type: HISTORICAL ({len(request['bars'])} bars)")

        # Convert to DataFrame
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

        print("\nDataFrame Info:")
        print(df.info())
        print("\nDataFrame Head:")
        print(df.head(10))
        print("\nDataFrame Tail:")
        print(df.tail(10))
        print("\nDataFrame Stats:")
        print(df.describe())

        # Store historical data
        historical_data = df.copy()

        # Train the model on historical data
        trading_model.train(df, epochs=50)

        print("="*50)
        print(f"DataFrame Shape: {df.shape}")
        print("="*50 + "\n")

        return {
            "status": "ok",
            "bars_received": len(df),
            "data_type": request.get('type', 'unknown'),
            "model_trained": True
        }

    else:
        # Realtime data - single bar
        print(f"Data Type: REALTIME (single bar)")

        # Convert to DataFrame
        new_bar = pd.DataFrame([{
            'time': request['time'],
            'open': request['open'],
            'high': request['high'],
            'low': request['low'],
            'close': request['close']
        }])

        # Convert time to datetime
        new_bar['time'] = pd.to_datetime(new_bar['time'])

        print("\nNew Bar Data:")
        print(new_bar.to_string(index=False))

        # Add new bar to historical data
        if historical_data is not None:
            historical_data = pd.concat([historical_data, new_bar], ignore_index=True)
        else:
            historical_data = new_bar.copy()

        # Make prediction
        signal, confidence = trading_model.predict(historical_data)

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

@app.get("/health-check")
def health_check():
    return {"status": "ok"}
