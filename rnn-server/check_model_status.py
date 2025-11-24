#!/usr/bin/env python3
"""
Quick Model Status Check (No server needed)

Checks the model file directly without needing the server running.
"""

import torch
from pathlib import Path

print("="*70)
print("MODEL STATUS CHECK")
print("="*70)

model_path = Path("models/trading_model.pth")

# Check if model file exists
print(f"\n1. Model file exists: {model_path.exists()}")

if model_path.exists():
    print(f"   Path: {model_path}")
    print(f"   Size: {model_path.stat().st_size / 1024:.1f} KB")
    
    from datetime import datetime
    mod_time = datetime.fromtimestamp(model_path.stat().st_mtime)
    print(f"   Last modified: {mod_time}")
    
    # Load and check checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"\n2. Model checkpoint loaded successfully")
        print(f"   is_trained flag: {checkpoint.get('is_trained', 'NOT FOUND')}")
        print(f"   sequence_length: {checkpoint.get('sequence_length', 'NOT FOUND')}")
        print(f"   signal_threshold: {checkpoint.get('signal_threshold', 'NOT FOUND')}")
        
        # Check if scaler is present
        has_scaler = checkpoint.get('scaler_mean') is not None
        print(f"   Has scaler: {has_scaler}")
        
        # Try loading the model
        print(f"\n3. Testing model load...")
        from model import TradingModel
        
        model = TradingModel(sequence_length=15)
        print(f"   Model created")
        print(f"   is_trained (before load): {model.is_trained}")
        
        success = model.load_model()
        print(f"   Load successful: {success}")
        print(f"   is_trained (after load): {model.is_trained}")
        
        print(f"\n{'='*70}")
        print("DIAGNOSIS:")
        print(f"{'='*70}")
        
        if model.is_trained:
            print(" Model is properly trained and loads correctly!")
            print("\nThe issue is likely:")
            print("  - Server not running")
            print("  - Server didn't load model on startup")
            print("  - Training still in progress")
            print("\nSOLUTION:")
            print("  Start/restart the server:")
            print("    cd rnn-server")
            print("    uv run fastapi dev main.py")
        else:
            print(" Model file exists but is_trained = False")
            print("\nThe checkpoint may be corrupted or incomplete.")
            print("\nSOLUTION:")
            print("  Retrain the model by sending historical data from NinjaTrader")
            
    except Exception as e:
        print(f"\n Error loading model: {e}")
        print("\nThe model file may be corrupted.")
        print("\nSOLUTION:")
        print("  Delete the model and retrain:")
        print("    rm models/trading_model.pth")
        print("    # Then send historical data from NinjaTrader to retrain")
        
else:
    print(f"\n Model file does not exist!")
    print(f"\nSOLUTION:")
    print("  Train the model by sending historical data from NinjaTrader")
    print("  The /analysis endpoint will trigger training automatically")

print(f"{'='*70}\n")
