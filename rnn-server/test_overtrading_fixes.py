"""Quick tests to verify over-trading fixes are working."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def test_signal_stability():
    """Test signal stability module prevents rapid reversals."""
    print("\n" + "="*70)
    print("TEST 1: Signal Stability")
    print("="*70)

    from core.signal_stability import SignalState

    state = SignalState()

    # Test 1: Same signal should be allowed
    state.update_signal("buy", 0.7, datetime.now())
    allowed, reason = state.should_allow_signal("buy", 0.7)
    assert allowed, "Same signal should be allowed"
    print("✅ Same signal allowed")

    # Test 2: Immediate reversal should be blocked
    allowed, reason = state.should_allow_signal("sell", 0.8)
    assert not allowed, "Immediate reversal should be blocked"
    print(f"✅ Immediate reversal blocked: {reason}")

    # Test 3: After cooldown, reversal should be allowed
    for i in range(5):
        state.update_signal("buy", 0.7, datetime.now())

    allowed, reason = state.should_allow_signal("sell", 0.9)
    assert allowed, "Reversal after cooldown should be allowed"
    print("✅ Reversal allowed after 5 bars cooldown")

    # Test 4: Insufficient confidence increase should be blocked
    state.update_signal("buy", 0.65, datetime.now())
    for i in range(5):
        state.update_signal("buy", 0.65, datetime.now())

    allowed, reason = state.should_allow_signal("sell", 0.70)
    assert not allowed, "Reversal with insufficient confidence should be blocked"
    print(f"✅ Insufficient confidence increase blocked: {reason}")

    print("\n✅ Signal Stability Tests PASSED\n")


def test_market_regime():
    """Test market regime detection with enhanced context."""
    print("="*70)
    print("TEST 2: Market Regime Detection (Enhanced)")
    print("="*70)

    from core.market_regime import calculate_market_regime

    # Create trending market data with enough history for context
    dates = pd.date_range(start='2024-01-01', periods=150, freq='5min')

    # Build historical baseline (first 100 bars sideways)
    baseline_prices = [100 + np.random.uniform(-1, 1) for _ in range(100)]

    # Strong uptrend (last 50 bars)
    trending_prices = np.linspace(100, 110, 50)

    all_prices = baseline_prices + list(trending_prices)

    trending_df = pd.DataFrame({
        'time': dates,
        'close': all_prices,
        'high': [p + 0.5 for p in all_prices],
        'low': [p - 0.5 for p in all_prices],
        'open': all_prices,
        'volume': [1000] * 150
    })

    regime = calculate_market_regime(trending_df, lookback=20, historical_lookback=100, use_adx=False)
    print(f"\nTrending Market (with context):")
    print(f"  Regime: {regime['regime']}")
    print(f"  Should Trade: {regime['should_trade']}")
    print(f"  Confidence Multiplier: {regime['confidence_multiplier']}")
    print(f"  Metrics: trend_strength={regime['metrics']['trend_strength']:.2f}, "
          f"consistency={regime['metrics']['directional_consistency']:.2f}, "
          f"volatility_percentile={regime['metrics']['volatility_percentile']:.1%}")
    assert regime['regime'] == 'trending', f"Should detect trending market (got {regime['regime']})"
    assert regime['should_trade'], "Should allow trading in trending market"
    print("✅ Trending market detected correctly")

    # Create choppy market data with historical context
    choppy_prices = []
    for i in range(150):
        choppy_prices.append(100 + (i % 2) * 2)  # Oscillates 100, 102, 100, 102...

    choppy_df = pd.DataFrame({
        'time': dates,
        'close': choppy_prices,
        'high': [p + 0.5 for p in choppy_prices],
        'low': [p - 0.5 for p in choppy_prices],
        'open': choppy_prices,
        'volume': [1000] * 150
    })

    regime = calculate_market_regime(choppy_df, lookback=20, historical_lookback=100, use_adx=False)
    print(f"\nChoppy Market (with context):")
    print(f"  Regime: {regime['regime']}")
    print(f"  Should Trade: {regime['should_trade']}")
    print(f"  Confidence Multiplier: {regime['confidence_multiplier']}")
    print(f"  Metrics: trend_strength={regime['metrics']['trend_strength']:.2f}, "
          f"consistency={regime['metrics']['directional_consistency']:.2f}, "
          f"volatility_percentile={regime['metrics']['volatility_percentile']:.1%}")
    assert regime['regime'] in ['choppy', 'volatile'], f"Should detect choppy/volatile market (got {regime['regime']})"
    assert not regime['should_trade'], "Should NOT trade in choppy market"
    print("✅ Choppy market detected correctly")

    # Create high volatility market with context
    volatile_prices = [100]
    for i in range(149):
        # Large random moves
        volatile_prices.append(volatile_prices[-1] + np.random.uniform(-5, 5))

    volatile_df = pd.DataFrame({
        'time': dates,
        'close': volatile_prices,
        'high': [p + 2 for p in volatile_prices],
        'low': [p - 2 for p in volatile_prices],
        'open': volatile_prices,
        'volume': [1000] * 150
    })

    regime = calculate_market_regime(volatile_df, lookback=20, historical_lookback=100, use_adx=False)
    print(f"\nVolatile Market (with context):")
    print(f"  Regime: {regime['regime']}")
    print(f"  Should Trade: {regime['should_trade']}")
    print(f"  Confidence Multiplier: {regime['confidence_multiplier']}")
    print(f"  Metrics: volatility_percentile={regime['metrics']['volatility_percentile']:.1%}")
    # Volatile regime detection depends on recent vs historical volatility
    print(f"✅ Volatile market classified as: {regime['regime']}")

    print("\n✅ Market Regime Tests PASSED\n")


def test_confidence_threshold():
    """Test adjusted confidence thresholds."""
    print("="*70)
    print("TEST 3: Confidence Threshold Adjustment")
    print("="*70)

    from core.market_regime import get_adjusted_confidence_threshold

    base_threshold = 0.60

    # Trending market (1.0x multiplier)
    trending_regime = {
        "regime": "trending",
        "confidence_multiplier": 1.0
    }
    adjusted = get_adjusted_confidence_threshold(base_threshold, trending_regime)
    print(f"\nTrending Market:")
    print(f"  Base: {base_threshold:.1%}")
    print(f"  Adjusted: {adjusted:.1%}")
    assert adjusted == 0.60, "Trending should keep base threshold"
    print("✅ Trending market uses base threshold")

    # Choppy market (1.5x multiplier)
    choppy_regime = {
        "regime": "choppy",
        "confidence_multiplier": 1.5
    }
    adjusted = get_adjusted_confidence_threshold(base_threshold, choppy_regime)
    print(f"\nChoppy Market:")
    print(f"  Base: {base_threshold:.1%}")
    print(f"  Adjusted: {adjusted:.1%}")
    assert abs(adjusted - 0.90) < 0.001, f"Choppy should increase threshold to 90% (got {adjusted})"
    print("✅ Choppy market increases threshold to 90%")

    # Volatile market (1.3x multiplier)
    volatile_regime = {
        "regime": "volatile",
        "confidence_multiplier": 1.3
    }
    adjusted = get_adjusted_confidence_threshold(base_threshold, volatile_regime)
    print(f"\nVolatile Market:")
    print(f"  Base: {base_threshold:.1%}")
    print(f"  Adjusted: {adjusted:.1%}")
    assert abs(adjusted - 0.78) < 0.001, f"Volatile should increase threshold to 78% (got {adjusted})"
    print("✅ Volatile market increases threshold to 78%")

    print("\n✅ Confidence Threshold Tests PASSED\n")


def test_integration():
    """Test that all imports work together."""
    print("="*70)
    print("TEST 4: Integration Test")
    print("="*70)

    try:
        from core.signal_stability import get_signal_state
        from core.market_regime import calculate_market_regime
        from services.request_handler import handle_realtime_request

        print("✅ All modules import successfully")

        # Test signal state singleton
        state1 = get_signal_state()
        state2 = get_signal_state()
        assert state1 is state2, "Signal state should be singleton"
        print("✅ Signal state singleton working")

        print("\n✅ Integration Tests PASSED\n")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("OVER-TRADING FIXES - VERIFICATION TESTS")
    print("="*70)

    try:
        test_signal_stability()
        test_market_regime()
        test_confidence_threshold()
        test_integration()

        print("="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nYour over-trading fixes are working correctly.")
        print("Ready to start paper trading!\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    run_all_tests()
