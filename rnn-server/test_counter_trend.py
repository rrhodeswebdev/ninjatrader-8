"""
Test Script for Counter-Trend Trading Improvements

This script tests the counter-trend filtering and target adjustment logic
with various market scenarios.

Usage:
    cd rnn-server
    uv run python test_counter_trend.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from model import detect_market_regime_enhanced, calculate_trend_alignment_feature
from risk_management import CounterTrendFilter, StopTargetCalculator, RiskManager


def create_test_scenario(scenario_type='trending_bullish', bars=200):
    """
    Create synthetic market data for different scenarios

    Args:
        scenario_type: One of 'trending_bullish', 'trending_bearish', 'ranging', 'choppy'
        bars: Number of bars to generate

    Returns:
        DataFrame with OHLCV data
    """
    base_price = 4500.0
    dates = [datetime.now() - timedelta(minutes=bars-i) for i in range(bars)]

    if scenario_type == 'trending_bullish':
        # Strong uptrend with ADX > 25
        close_prices = base_price + np.cumsum(np.random.normal(0.5, 1.5, bars))
        high_prices = close_prices + np.random.uniform(1, 3, bars)
        low_prices = close_prices - np.random.uniform(1, 3, bars)
        open_prices = low_prices + (high_prices - low_prices) * np.random.uniform(0.3, 0.7, bars)

    elif scenario_type == 'trending_bearish':
        # Strong downtrend with ADX > 25
        close_prices = base_price - np.cumsum(np.random.normal(0.5, 1.5, bars))
        high_prices = close_prices + np.random.uniform(1, 3, bars)
        low_prices = close_prices - np.random.uniform(1, 3, bars)
        open_prices = low_prices + (high_prices - low_prices) * np.random.uniform(0.3, 0.7, bars)

    elif scenario_type == 'ranging':
        # Ranging market with low ADX
        close_prices = base_price + np.sin(np.linspace(0, 4*np.pi, bars)) * 10 + np.random.normal(0, 1, bars)
        high_prices = close_prices + np.random.uniform(1, 3, bars)
        low_prices = close_prices - np.random.uniform(1, 3, bars)
        open_prices = low_prices + (high_prices - low_prices) * np.random.uniform(0.3, 0.7, bars)

    else:  # choppy
        # High volatility, no clear trend
        close_prices = base_price + np.cumsum(np.random.normal(0, 3, bars))
        high_prices = close_prices + np.random.uniform(2, 8, bars)
        low_prices = close_prices - np.random.uniform(2, 8, bars)
        open_prices = low_prices + (high_prices - low_prices) * np.random.uniform(0.3, 0.7, bars)

    df = pd.DataFrame({
        'time': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(1000, 5000, bars)
    })

    return df


def test_regime_detection():
    """Test enhanced regime detection"""
    print("\n" + "="*70)
    print("TEST 1: Enhanced Regime Detection")
    print("="*70)

    scenarios = ['trending_bullish', 'trending_bearish', 'ranging', 'choppy']

    for scenario in scenarios:
        df = create_test_scenario(scenario)
        regime_info = detect_market_regime_enhanced(df)

        print(f"\n{scenario.upper()} Market:")
        print(f"  Regime: {regime_info['regime']}")
        print(f"  Trend Direction: {regime_info['trend_direction']}")
        print(f"  Trend Strength (ADX): {regime_info['trend_strength']:.2f}")
        print(f"  Volatility Ratio: {regime_info['vol_ratio']:.2f}")


def test_counter_trend_filter():
    """Test counter-trend filtering logic"""
    print("\n" + "="*70)
    print("TEST 2: Counter-Trend Signal Filtering")
    print("="*70)

    filter = CounterTrendFilter(
        enable_filtering=True,
        block_counter_trends_in_strong_trends=True
    )

    test_cases = [
        {
            'name': 'LONG signal in BULLISH trend (with-trend)',
            'signal': 'long',
            'confidence': 0.75,
            'regime_info': {
                'regime': 'trending_normal',
                'trend_direction': 'bullish',
                'trend_strength': 30.0
            }
        },
        {
            'name': 'SHORT signal in BULLISH trend (counter-trend)',
            'signal': 'short',
            'confidence': 0.75,
            'regime_info': {
                'regime': 'trending_normal',
                'trend_direction': 'bullish',
                'trend_strength': 30.0
            }
        },
        {
            'name': 'LONG signal in BEARISH trend (counter-trend)',
            'signal': 'long',
            'confidence': 0.75,
            'regime_info': {
                'regime': 'trending_normal',
                'trend_direction': 'bearish',
                'trend_strength': 28.0
            }
        },
        {
            'name': 'SHORT signal in RANGING market (counter-trend OK)',
            'signal': 'short',
            'confidence': 0.70,
            'regime_info': {
                'regime': 'ranging_normal',
                'trend_direction': 'bullish',
                'trend_strength': 15.0
            }
        },
    ]

    for test in test_cases:
        print(f"\n{test['name']}:")
        print(f"  Input: {test['signal'].upper()} @ {test['confidence']:.0%} confidence")

        filtered_signal, filtered_conf, details = filter.filter_signal(
            test['signal'],
            test['confidence'],
            test['regime_info']
        )

        print(f"  Output: {filtered_signal.upper()} @ {filtered_conf:.0%} confidence")
        print(f"  Filtered: {details.get('filtered', False)}")
        print(f"  Reason: {details['reason']}")


def test_target_adjustment():
    """Test target adjustment for counter-trend trades"""
    print("\n" + "="*70)
    print("TEST 3: Target Adjustment for Counter-Trend Trades")
    print("="*70)

    calc = StopTargetCalculator()

    test_cases = [
        {
            'name': 'LONG in strong BULLISH trend (with-trend)',
            'entry_price': 4500.0,
            'direction': 'long',
            'atr': 15.0,
            'regime': 'trending_normal',
            'trend_direction': 'bullish',
            'trend_strength': 30.0
        },
        {
            'name': 'SHORT in strong BULLISH trend (counter-trend)',
            'entry_price': 4500.0,
            'direction': 'short',
            'atr': 15.0,
            'regime': 'trending_normal',
            'trend_direction': 'bullish',
            'trend_strength': 30.0
        },
        {
            'name': 'SHORT in weak BULLISH trend (counter-trend, mild)',
            'entry_price': 4500.0,
            'direction': 'short',
            'atr': 15.0,
            'regime': 'ranging_normal',
            'trend_direction': 'bullish',
            'trend_strength': 18.0
        },
    ]

    for test in test_cases:
        print(f"\n{test['name']}:")
        stops = calc.calculate_stops(
            entry_price=test['entry_price'],
            direction=test['direction'],
            atr=test['atr'],
            regime=test['regime'],
            trend_direction=test['trend_direction'],
            trend_strength=test['trend_strength']
        )

        print(f"  Stop Loss: ${stops['stop_loss']:.2f} ({stops['stop_distance']:.2f} points)")
        print(f"  Take Profit: ${stops['take_profit']:.2f} ({stops['target_distance']:.2f} points)")
        print(f"  Risk/Reward: {stops['risk_reward']:.2f}")
        print(f"  Counter-Trend: {stops['is_counter_trend']}")
        if stops['is_counter_trend']:
            print(f"  Target Adjustment: {stops['target_adjustment']:.0%}")


def test_full_integration():
    """Test complete integration through RiskManager"""
    print("\n" + "="*70)
    print("TEST 4: Full Integration Test")
    print("="*70)

    risk_mgr = RiskManager()

    # Create a trending bullish market
    df = create_test_scenario('trending_bullish', bars=200)
    regime_info = detect_market_regime_enhanced(df)

    print(f"\nMarket State:")
    print(f"  Regime: {regime_info['regime']}")
    print(f"  Trend: {regime_info['trend_direction'].upper()} (ADX={regime_info['trend_strength']:.1f})")

    # Test counter-trend SHORT signal (should be blocked or penalized)
    print(f"\n--- Counter-Trend Trade (SHORT in BULLISH market) ---")
    trade_params = risk_mgr.calculate_trade_parameters(
        signal='short',
        confidence=0.80,
        entry_price=df['close'].iloc[-1],
        atr=15.0,
        regime=regime_info['regime'],
        account_balance=25000.0,
        regime_info=regime_info
    )

    print(f"Signal: {trade_params['signal'].upper()}")
    if trade_params['signal'] != 'hold':
        print(f"Confidence: {trade_params['confidence']:.2%} (was {trade_params.get('original_confidence', 0):.2%})")
        print(f"Contracts: {trade_params['contracts']}")
    else:
        print(f"Confidence: 0% (blocked)")
        print(f"Contracts: 0")
    if 'filter_details' in trade_params:
        print(f"Filter Reason: {trade_params['filter_details'].get('reason', 'N/A')}")
    elif 'reason' in trade_params:
        print(f"Reason: {trade_params['reason']}")

    # Test with-trend LONG signal (should be allowed)
    print(f"\n--- With-Trend Trade (LONG in BULLISH market) ---")
    trade_params = risk_mgr.calculate_trade_parameters(
        signal='long',
        confidence=0.75,
        entry_price=df['close'].iloc[-1],
        atr=15.0,
        regime=regime_info['regime'],
        account_balance=25000.0,
        regime_info=regime_info
    )

    print(f"Signal: {trade_params['signal'].upper()}")
    print(f"Confidence: {trade_params['confidence']:.2%}")
    print(f"Contracts: {trade_params['contracts']}")
    print(f"Entry: ${trade_params['entry_price']:.2f}")
    print(f"Stop: ${trade_params['stop_loss']:.2f}")
    print(f"Target: ${trade_params['take_profit']:.2f}")
    print(f"R:R: {trade_params['risk_reward']:.2f}")


def test_trend_alignment_feature():
    """Test trend alignment feature calculation"""
    print("\n" + "="*70)
    print("TEST 5: Trend Alignment Feature")
    print("="*70)

    scenarios = ['trending_bullish', 'trending_bearish', 'ranging']

    for scenario in scenarios:
        df = create_test_scenario(scenario)
        alignment = calculate_trend_alignment_feature(df)

        print(f"\n{scenario.upper()}:")
        print(f"  Recent alignment values:")
        print(f"    {alignment.tail(5).values}")
        print(f"  Mean: {alignment.mean():.4f}")
        print(f"  Current: {alignment.iloc[-1]:.4f}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("COUNTER-TREND TRADING IMPROVEMENTS - TEST SUITE")
    print("="*70)

    try:
        test_regime_detection()
        test_counter_trend_filter()
        test_target_adjustment()
        test_full_integration()
        test_trend_alignment_feature()

        print("\n" + "="*70)
        print(" ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
