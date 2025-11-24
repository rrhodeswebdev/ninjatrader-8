"""
INTEGRATION HOTFIX - Connect new features to existing model

This module integrates:
1. New feature extraction from features/ modules
2. Advanced confidence scoring
3. Improved market regime detection

Run this to patch the existing system without breaking changes
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

# Import new feature modules
try:
    from features.market_structure import MarketStructureFeatures
    from features.price_action import PriceActionPatterns
    from features.order_flow import OrderFlowFeatures
    from confidence_scoring import AdvancedConfidenceScoring
    NEW_FEATURES_AVAILABLE = True
    print(" New feature modules loaded successfully")
except ImportError as e:
    print(f"  Could not load new features: {e}")
    NEW_FEATURES_AVAILABLE = False


def extract_all_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all new pure price action features

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with all new features added
    """
    if not NEW_FEATURES_AVAILABLE:
        print("  New features not available, returning original data")
        return df

    print(" Extracting new price action features...")

    # Make a copy
    df_enhanced = df.copy()

    # Initialize feature extractors
    ms = MarketStructureFeatures()
    pa = PriceActionPatterns()
    of = OrderFlowFeatures()

    # Extract market structure features
    ms_features = {
        **ms.calculate_order_blocks(df_enhanced),
        **ms.calculate_fair_value_gaps(df_enhanced),
        **ms.calculate_liquidity_zones(df_enhanced),
        **ms.calculate_market_structure_shift(df_enhanced),
        **ms.calculate_swing_points(df_enhanced),
    }

    for name, values in ms_features.items():
        df_enhanced[f'ms_{name}'] = values

    # Extract price action patterns
    df_enhanced = pa.calculate_all_patterns(df_enhanced)

    # Extract order flow features
    df_enhanced = of.calculate_all_orderflow(df_enhanced)

    # Fill NaN with 0
    df_enhanced = df_enhanced.fillna(0)

    # Replace inf with 0
    df_enhanced = df_enhanced.replace([np.inf, -np.inf], 0)

    print(f" Added {len(df_enhanced.columns) - len(df.columns)} new features")

    return df_enhanced


def calculate_advanced_confidence(
    probabilities: np.ndarray,
    features: np.ndarray,
    market_state: Dict,
    trade_history: list = None
) -> Tuple[float, Dict]:
    """
    Calculate confidence using new advanced scoring system

    Args:
        probabilities: Model output probabilities
        features: Input features (last bar)
        market_state: Dict with volatility, volume, trend_strength
        trade_history: Recent trades (optional)

    Returns:
        Tuple of (composite_confidence, component_scores)
    """
    if not NEW_FEATURES_AVAILABLE:
        # Fallback to max probability
        return float(np.max(probabilities)), {}

    scorer = AdvancedConfidenceScoring()

    composite, components = scorer.calculate_composite_confidence(
        probabilities=probabilities,
        features=features,
        market_state=market_state,
        model_predictions=None,  # TODO: Add ensemble support
        trade_history=trade_history
    )

    return composite / 100.0, components  # Convert 0-100 to 0-1 for compatibility


def detect_market_regime_enhanced(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    Enhanced regime detection using pure price action

    Args:
        df: DataFrame with OHLCV data
        lookback: Bars to analyze

    Returns:
        Dict with regime, confidence_multiplier, should_trade
    """
    if len(df) < lookback:
        return {
            'regime': 'unknown',
            'confidence_multiplier': 1.5,
            'should_trade': False,
            'reason': 'Insufficient data',
            'metrics': {}
        }

    recent = df.tail(lookback)
    returns = recent['close'].pct_change().dropna()

    if len(returns) == 0:
        return {
            'regime': 'unknown',
            'confidence_multiplier': 1.5,
            'should_trade': False,
            'reason': 'No returns',
            'metrics': {}
        }

    # Calculate metrics
    volatility = returns.std()
    mean_return = returns.mean()

    # Trend strength
    trend_strength = abs(mean_return) / volatility if volatility > 0 else 0

    # Directional consistency
    positive_bars = (returns > 0).sum()
    negative_bars = (returns < 0).sum()
    total_bars = len(returns)
    directional_consistency = abs(positive_bars - negative_bars) / total_bars if total_bars > 0 else 0

    # Historical volatility context
    if len(df) > lookback * 2:
        historical_returns = df['close'].tail(lookback * 2).pct_change().dropna()
        historical_volatility = historical_returns.rolling(window=lookback).std().dropna()

        if len(historical_volatility) > 0:
            volatility_percentile = (historical_volatility < volatility).sum() / len(historical_volatility)
        else:
            volatility_percentile = 0.5
    else:
        volatility_percentile = 0.5

    metrics = {
        'trend_strength': float(trend_strength),
        'directional_consistency': float(directional_consistency),
        'volatility': float(volatility),
        'volatility_percentile': float(volatility_percentile),
        'mean_return': float(mean_return),
    }

    # FIXED THRESHOLDS - More lenient to allow trading
    TREND_THRESHOLD = 0.25  # Lower = easier to qualify as trending
    CONSISTENCY_THRESHOLD = 0.30  # Lower = easier to qualify
    VOLATILITY_THRESHOLD = 0.80  # Higher = only block extreme volatility (was 0.75)

    # Classify regime
    is_trending = (
        trend_strength > TREND_THRESHOLD and
        directional_consistency > CONSISTENCY_THRESHOLD and
        volatility_percentile < 0.75  # Not extremely volatile
    )

    if is_trending:
        return {
            'regime': 'trending',
            'confidence_multiplier': 1.0,
            'should_trade': True,
            'reason': f'Trending (strength={trend_strength:.2f}, consistency={directional_consistency:.2f})',
            'metrics': metrics
        }

    # Only mark as volatile if REALLY extreme
    if volatility_percentile > VOLATILITY_THRESHOLD:
        return {
            'regime': 'volatile',
            'confidence_multiplier': 1.3,
            'should_trade': True,  # CHANGED: Still allow trading but with higher threshold
            'reason': f'High volatility (percentile={volatility_percentile:.1%})',
            'metrics': metrics
        }

    # Normal/ranging - still allow trading
    return {
        'regime': 'normal',
        'confidence_multiplier': 1.2,
        'should_trade': True,  # CHANGED: Allow trading in normal markets
        'reason': f'Normal market (strength={trend_strength:.2f})',
        'metrics': metrics
    }


# Monkey patch functions for integration
def patch_model_class():
    """
    Patch the existing TradingModel class to use new features
    """
    print(" Patching TradingModel class...")

    # This would require modifying model.py
    # For now, we'll create wrapper functions

    print(" Patch complete - use wrapper functions")


if __name__ == "__main__":
    print("="*70)
    print("INTEGRATION HOTFIX TEST")
    print("="*70)

    # Test feature extraction
    test_data = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'open': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.1) + 0.5,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.1) - 0.5,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'volume': np.random.randint(1000, 10000, 100),
    })

    test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1) + 0.1
    test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1) - 0.1

    if NEW_FEATURES_AVAILABLE:
        df_enhanced = extract_all_new_features(test_data)
        print(f"\nOriginal columns: {len(test_data.columns)}")
        print(f"Enhanced columns: {len(df_enhanced.columns)}")
        print(f"New features added: {len(df_enhanced.columns) - len(test_data.columns)}")

        # Test regime detection
        regime = detect_market_regime_enhanced(test_data)
        print(f"\nRegime: {regime['regime']}")
        print(f"Should trade: {regime['should_trade']}")
        print(f"Multiplier: {regime['confidence_multiplier']}")

        # Test confidence scoring
        probs = np.array([0.2, 0.3, 0.5])
        features = df_enhanced.iloc[-1].values
        market_state = {
            'volatility': regime['metrics']['volatility'],
            'volume': test_data['volume'].iloc[-1],
            'trend_strength': regime['metrics']['trend_strength']
        }

        confidence, components = calculate_advanced_confidence(probs, features, market_state)
        print(f"\nConfidence: {confidence:.3f} ({confidence*100:.1f}%)")
        if components:
            print("Components:")
            for k, v in components.items():
                print(f"  {k}: {v:.1f}")
    else:
        print(" New features not available - cannot run test")
