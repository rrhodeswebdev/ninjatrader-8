#!/usr/bin/env python3
"""
Test complete request flow from NinjaTrader to predictions
"""

import json
import urllib.request
import urllib.error
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_endpoint(name, url, method='GET', data=None):
    """Test an endpoint and show detailed results"""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    print(f"URL: {url}")
    print(f"Method: {method}")

    try:
        if data:
            print(f"Request data: {json.dumps(data, indent=2)}")
            data_bytes = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={'Content-Type': 'application/json'},
                method=method
            )
        else:
            req = urllib.request.Request(url, method=method)

        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode('utf-8'))
            print(f"\n✓ SUCCESS")
            print(f"Response: {json.dumps(result, indent=2)}")
            return True, result

    except urllib.error.HTTPError as e:
        print(f"\n✗ HTTP ERROR {e.code}")
        try:
            error_body = e.read().decode('utf-8')
            print(f"Error body: {error_body}")
        except:
            pass
        return False, None

    except urllib.error.URLError as e:
        print(f"\n✗ CONNECTION ERROR")
        print(f"Error: {e.reason}")
        print("\n⚠️  Server is not running!")
        print("Start server with: cd rnn-server && uv run fastapi dev main.py")
        return False, None

    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR")
        print(f"Error: {e}")
        return False, None

def main():
    print("="*70)
    print("COMPREHENSIVE REQUEST FLOW TEST")
    print("="*70)

    # Test 1: Server is running
    success, _ = test_endpoint(
        "Server Health",
        f"{BASE_URL}/health-check"
    )

    if not success:
        print("\n" + "="*70)
        print("STOPPED: Server is not running")
        print("="*70)
        return

    # Test 2: Training status
    test_endpoint(
        "Training Status",
        f"{BASE_URL}/training-status"
    )

    # Test 3: Realtime prediction (minimal data)
    test_endpoint(
        "Realtime Prediction - Minimal",
        f"{BASE_URL}/analysis",
        method='POST',
        data={
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
    )

    # Test 4: Realtime prediction with position info
    test_endpoint(
        "Realtime Prediction - With Position",
        f"{BASE_URL}/analysis",
        method='POST',
        data={
            "type": "realtime",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "open": 4500.0,
            "high": 4505.0,
            "low": 4498.0,
            "close": 4502.0,
            "volume": 1000.0,
            "dailyPnL": 0.0,
            "dailyMaxLoss": 250.0,
            "current_position": "flat",
            "entry_price": 0.0,
            "position_quantity": 0
        }
    )

    # Test 5: With daily loss hit
    test_endpoint(
        "Daily Max Loss Hit",
        f"{BASE_URL}/analysis",
        method='POST',
        data={
            "type": "realtime",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "open": 4500.0,
            "high": 4505.0,
            "low": 4498.0,
            "close": 4502.0,
            "volume": 1000.0,
            "dailyPnL": -260.0,  # Over the limit
            "dailyMaxLoss": 250.0
        }
    )

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("\nIf all tests passed, the server is working correctly.")
    print("\nNow check:")
    print("1. Is NinjaTrader sending requests to http://localhost:8000/analysis ?")
    print("2. Check the server console for 'RECEIVED DATA AT /analysis' messages")
    print("3. Verify the NinjaTrader strategy has the correct server URL")
    print("\nIf you don't see 'RECEIVED DATA' in server console when bars update,")
    print("then NinjaTrader is NOT sending requests.")
    print("="*70)

if __name__ == "__main__":
    main()
