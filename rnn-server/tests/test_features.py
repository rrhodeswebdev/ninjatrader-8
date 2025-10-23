"""
Unit Tests for Feature Modules
"""

import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '..')

from features.market_structure import MarketStructureFeatures
from features.price_action import PriceActionPatterns
from features.order_flow import OrderFlowFeatures
from features.multi_timeframe import MultiTimeframeFeatures


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data"""
    np.random.seed(42)
    n = 100

    data = {
        'time': pd.date_range('2024-01-01', periods=n, freq='1min'),
        'open': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(n) * 0.1) + 0.5,
        'low': 100 + np.cumsum(np.random.randn(n) * 0.1) - 0.5,
        'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'volume': np.random.randint(1000, 10000, n),
        'bid_volume': np.random.randint(400, 5000, n),
        'ask_volume': np.random.randint(400, 5000, n),
    }

    df = pd.DataFrame(data)
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'close']].max(axis=1) + 0.1
    df['low'] = df[['open', 'low', 'close']].min(axis=1) - 0.1

    return df


def test_market_structure_order_blocks(sample_ohlcv):
    """Test order block detection"""
    ms = MarketStructureFeatures()
    result = ms.calculate_order_blocks(sample_ohlcv)

    assert 'ob_bullish_high' in result
    assert 'ob_bearish_high' in result
    assert len(result['ob_strength']) == len(sample_ohlcv)


def test_market_structure_fvg(sample_ohlcv):
    """Test FVG detection"""
    ms = MarketStructureFeatures()
    result = ms.calculate_fair_value_gaps(sample_ohlcv)

    assert 'fvg_bullish_active' in result
    assert 'fvg_bearish_active' in result
    assert 'fvg_size' in result


def test_price_action_engulfing(sample_ohlcv):
    """Test engulfing pattern detection"""
    pa = PriceActionPatterns()
    result = pa.detect_engulfing_patterns(sample_ohlcv)

    assert 'bullish_engulfing' in result
    assert 'bearish_engulfing' in result
    assert 'engulfing_strength' in result


def test_price_action_pin_bars(sample_ohlcv):
    """Test pin bar detection"""
    pa = PriceActionPatterns()
    result = pa.detect_pin_bars(sample_ohlcv)

    assert 'bullish_pin' in result
    assert 'bearish_pin' in result
    assert 'pin_wick_size' in result


def test_order_flow_volume_profile(sample_ohlcv):
    """Test volume profile calculation"""
    of = OrderFlowFeatures()
    result = of.calculate_volume_profile(sample_ohlcv)

    assert 'poc_price' in result
    assert 'value_area_high' in result
    assert 'value_area_low' in result


def test_order_flow_delta(sample_ohlcv):
    """Test delta analysis"""
    of = OrderFlowFeatures()
    result = of.calculate_delta_analysis(sample_ohlcv)

    assert 'cumulative_delta' in result
    assert 'delta_divergence' in result
    assert 'aggressive_buyers' in result


def test_integration_all_features(sample_ohlcv):
    """Test integration of all features"""
    ms = MarketStructureFeatures()
    pa = PriceActionPatterns()
    of = OrderFlowFeatures()

    # Calculate all features
    df = sample_ohlcv.copy()
    df_ms = ms.calculate_order_blocks(df)
    df_pa = pa.calculate_all_patterns(df)
    df_of = of.calculate_all_orderflow(df)

    # Verify no errors and features added
    assert len(df_pa) == len(sample_ohlcv)
    assert len(df_of) == len(sample_ohlcv)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
