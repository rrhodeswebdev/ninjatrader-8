#!/usr/bin/env python3
"""
Comprehensive Prediction Diagnostics (No external dependencies)

This script tests all the checks that could block predictions and identifies
exactly where the problem is.
"""

import json
import urllib.request
import urllib.error
from datetime import datetime

BASE_URL = "http://localhost:8000"

def make_request(url, method='GET', data=None):
    """Make HTTP request without requests library"""
    try:
        if data:
            data = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method=method)
        else:
            req = urllib.request.Request(url, method=method)
        
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError as e:
        return None
    except Exception as e:
        print(f"Request error: {e}")
        return None

def check_server():
    """Check if server is running"""
    result = make_request(f"{BASE_URL}/")
    return result is not None

def check_health():
    """Check health endpoint"""
    data = make_request(f"{BASE_URL}/health-check")
    if data:
        print(f"✓ Health check response:")
        print(f"  Status: {data.get('status')}")
        print(f"  Model trained: {data.get('model_trained')}")
        print(f"  Device: {data.get('device')}")
    else:
        print(f"✗ Health check failed")
    return data

def check_training_status():
    """Check training status"""
    data = make_request(f"{BASE_URL}/training-status")
    if data:
        print(f"\n✓ Training status:")
        print(f"  Is training: {data.get('is_training')}")
        print(f"  Progress: {data.get('progress')}")
        print(f"  Error: {data.get('error')}")
    else:
        print(f"\n✗ Training status check failed")
    return data

def test_prediction(scenario_name, **params):
    """Test a prediction with given parameters"""
    print(f"\n{'='*70}")
    print(f"Testing: {scenario_name}")
    print(f"{'='*70}")

    # Default request
    request_data = {
        "type": "realtime",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "open": 4500.0,
        "high": 4505.0,
        "low": 4498.0,
        "close": 4502.0,
        "volume": 1000.0,
        "dailyPnL": 0.0,
        "dailyMaxLoss": 250.0
    }

    # Update with custom params
    request_data.update(params)

    print(f"Request parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    data = make_request(f"{BASE_URL}/analysis", method='POST', data=request_data)
    
    if data:
        print(f"\nResponse:")
        print(f"  Status: {data.get('status')}")
        print(f"  Signal: {data.get('signal')}")
        print(f"  Confidence: {data.get('confidence')}")
        print(f"  Message: {data.get('message', 'N/A')}")

        # Additional info based on status
        if data.get('status') == 'training':
            print(f"  Training progress: {data.get('training_progress')}")
        elif data.get('status') == 'not_trained':
            print(f"  Training status: {data.get('training_status', 'N/A')}")
        elif data.get('status') == 'max_loss_hit':
            print(f"  Daily P&L: ${data.get('daily_pnl')}")
            print(f"  Max loss: ${data.get('daily_max_loss')}")
        elif data.get('status') == 'ok':
            print(f"  Raw signal: {data.get('raw_signal')}")
            print(f"  Filtered: {data.get('filtered')}")
            if data.get('risk_management'):
                rm = data['risk_management']
                print(f"  Contracts: {rm.get('contracts')}")
                print(f"  Entry: ${rm.get('entry_price')}")
    else:
        print(f"\n✗ Prediction request failed")

    return data

def main():
    print("="*70)
    print("COMPREHENSIVE PREDICTION DIAGNOSTICS")
    print("="*70)

    # Step 1: Check server
    print("\n1. Checking if server is running...")
    if not check_server():
        print("✗ Server is not running!")
        print("\n" + "="*70)
        print("SOLUTION:")
        print("="*70)
        print("Please start the server:")
        print("  cd rnn-server")
        print("  uv run fastapi dev main.py")
        print("\nThen run this diagnostic script again.")
        print("="*70)
        return
    print("✓ Server is running")

    # Step 2: Check health
    print("\n2. Checking server health...")
    health = check_health()
    if not health:
        print("✗ Health check failed - server may be starting up")
        print("Wait a few seconds and try again")
        return

    model_trained = health.get('model_trained')
    if not model_trained:
        print("\n⚠️  WARNING: Health check shows model_trained = False")
        print("This is the issue preventing predictions!")

    # Step 3: Check training status
    print("\n3. Checking training status...")
    training = check_training_status()
    if not training:
        print("✗ Training status check failed")
        return

    is_training = training.get('is_training')
    if is_training:
        print("\n⚠️  WARNING: Model is currently training")
        print("Predictions are blocked during training")
        print(f"Progress: {training.get('progress')}")

    # Step 4: Test various scenarios
    print("\n4. Testing prediction scenarios...")

    # Scenario 1: Normal request
    result1 = test_prediction(
        "Normal prediction request",
        dailyPnL=0.0,
        dailyMaxLoss=250.0
    )

    # Scenario 2: With small profit
    result2 = test_prediction(
        "Request with small profit",
        dailyPnL=50.0,
        dailyMaxLoss=250.0
    )

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    all_results = [result1, result2]
    statuses = [r.get('status') if r else 'ERROR' for r in all_results]

    print(f"\nTest results:")
    print(f"  Test 1 (Normal): {statuses[0]}")
    print(f"  Test 2 (With profit): {statuses[1]}")

    # Identify the issue
    print(f"\n{'='*70}")
    print("IDENTIFIED ISSUE & SOLUTION:")
    print(f"{'='*70}")
    
    if all(s == 'ok' for s in statuses):
        print("\n✅ All tests passed! Predictions are working correctly.")
        print("\nYour NinjaTrader bot should be receiving predictions.")
        
    elif 'training' in statuses:
        print("\n❌ ISSUE: Model is currently training")
        print("\nSOLUTION:")
        print("  Wait for training to complete.")
        print("  Monitor progress with:")
        print("    curl http://localhost:8000/training-status")
        
    elif 'not_trained' in statuses:
        print("\n❌ ISSUE: Model shows as not trained")
        print(f"   Model file exists: {health.get('model_trained')}")
        print(f"   Is training: {is_training}")
        print(f"   Training progress: {training.get('progress')}")
        print("\nSOLUTIONS (try in order):")
        print("\n  1. Reload the model:")
        print("     curl -X POST http://localhost:8000/load-model")
        print("\n  2. If that doesn't work, restart the server:")
        print("     # Press Ctrl+C to stop current server")
        print("     cd rnn-server")
        print("     uv run fastapi dev main.py")
        print("\n  3. If model was never trained, send historical data from NinjaTrader")
        
    elif 'max_loss_hit' in statuses:
        print("\n❌ ISSUE: Daily max loss limit has been hit")
        print("\nSOLUTION:")
        print("  This is actually working correctly!")
        print("  The bot is protecting you from further losses.")
        print("  To test, use dailyPnL=0 in your requests.")
        
    elif 'ERROR' in statuses:
        print("\n❌ ISSUE: Server errors during prediction")
        print("\nSOLUTION:")
        print("  Check the server terminal for error messages.")
        print("  Look for Python exceptions or tracebacks.")
        
    else:
        print(f"\n❓ ISSUE: Unknown problem")
        print(f"   Statuses received: {statuses}")
        print("\nSOLUTION:")
        print("  Check server logs for errors")
        print("  Look at the server terminal output")

    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
