#!/usr/bin/env python3
"""
Test Daily Max Loss Protection

This script tests that the daily max loss limit correctly blocks trading
when the loss threshold is exceeded.
"""

import requests
import json

# Server endpoint
BASE_URL = "http://localhost:8000"

def test_daily_max_loss():
    """Test daily max loss protection"""

    print("="*70)
    print("TESTING DAILY MAX LOSS PROTECTION")
    print("="*70)

    # Test scenarios
    test_cases = [
        {
            "name": "Normal Trading (No Loss)",
            "dailyPnL": 100.0,
            "dailyMaxLoss": 250.0,
            "expected_status": "ok",
            "should_trade": True
        },
        {
            "name": "Small Loss (Within Limit)",
            "dailyPnL": -100.0,
            "dailyMaxLoss": 250.0,
            "expected_status": "ok",
            "should_trade": True
        },
        {
            "name": "At Max Loss Threshold",
            "dailyPnL": -250.0,
            "dailyMaxLoss": 250.0,
            "expected_status": "max_loss_hit",
            "should_trade": False
        },
        {
            "name": "Exceeded Max Loss",
            "dailyPnL": -300.0,
            "dailyMaxLoss": 250.0,
            "expected_status": "max_loss_hit",
            "should_trade": False
        },
        {
            "name": "Large Loss Exceeded",
            "dailyPnL": -500.0,
            "dailyMaxLoss": 250.0,
            "expected_status": "max_loss_hit",
            "should_trade": False
        },
        {
            "name": "No Max Loss Set (Zero)",
            "dailyPnL": -500.0,
            "dailyMaxLoss": 0.0,
            "expected_status": "ok",  # Should trade if max loss is 0 (disabled)
            "should_trade": True
        }
    ]

    print("\nRunning test cases...\n")

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case['name']}")
        print(f"   Daily P&L: ${test_case['dailyPnL']:.2f}")
        print(f"   Max Loss: ${test_case['dailyMaxLoss']:.2f}")

        # Create test request
        request_data = {
            "type": "realtime",
            "time": "2025-01-15 10:30:00",
            "open": 4500.0,
            "high": 4505.0,
            "low": 4498.0,
            "close": 4502.0,
            "volume": 1000.0,
            "dailyPnL": test_case['dailyPnL'],
            "dailyMaxLoss": test_case['dailyMaxLoss']
        }

        try:
            response = requests.post(f"{BASE_URL}/analysis", json=request_data)
            result = response.json()

            status = result.get('status')
            signal = result.get('signal')

            # Check if test passed
            status_match = status == test_case['expected_status']
            signal_correct = (signal != 'hold') == test_case['should_trade']

            if status_match:
                print(f"    Status: {status}")
                passed += 1
            else:
                print(f"    Status: {status} (expected: {test_case['expected_status']})")
                failed += 1

            print(f"   Signal: {signal.upper()}")

            # Show additional info for max loss cases
            if status == "max_loss_hit":
                exceeded_by = result.get('exceeded_by', 0)
                print(f"   Exceeded by: ${exceeded_by:.2f}")
                print(f"   Message: {result.get('message')}")

        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1

        print()

    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {passed} ")
    print(f"Failed: {failed} ")

    if failed == 0:
        print("\n All tests passed! Daily max loss protection is working correctly.")
    else:
        print(f"\n  {failed} test(s) failed. Please review the implementation.")

    print("="*70)


def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health-check", timeout=2)
        return response.status_code == 200
    except:
        return False


if __name__ == "__main__":
    print("\nChecking if server is running...")

    if not check_server():
        print(" ERROR: Server is not running!")
        print("\nPlease start the server first:")
        print("  cd rnn-server")
        print("  uv run fastapi dev main.py")
        exit(1)

    print(" Server is running\n")

    test_daily_max_loss()
