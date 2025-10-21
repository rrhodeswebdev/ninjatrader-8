"""
Counter-Trend Trading Configuration

This file contains all configurable parameters for counter-trend trade filtering
and adjustment. Modify these values to tune the system's behavior.

Author: Claude Code
Date: 2025-10-20
"""

# =============================================================================
# COUNTER-TREND FILTER SETTINGS
# =============================================================================

COUNTER_TREND_FILTER_CONFIG = {
    # Enable/disable the entire counter-trend filtering system
    'enable_filtering': True,

    # ADX threshold for considering a market as "trending"
    # Values > 25 typically indicate a strong trend
    'trending_adx_threshold': 25.0,

    # How much to reduce confidence for counter-trend trades in trending markets
    # 0.5 = reduce by 50%
    'counter_trend_confidence_penalty': 0.5,

    # How much to boost confidence for counter-trend trades in ranging markets
    # 0.15 = boost by 15%
    'ranging_confidence_boost': 0.15,

    # Whether to completely block counter-trend trades in strong trends
    # True = block entirely, False = just reduce confidence
    'block_counter_trends_in_strong_trends': True,
}


# =============================================================================
# TARGET ADJUSTMENT SETTINGS
# =============================================================================

TARGET_ADJUSTMENT_CONFIG = {
    # Maximum percentage to reduce targets for counter-trend trades
    # 0.4 = up to 40% reduction in strong trends
    'max_target_reduction': 0.4,

    # ADX threshold where target reduction begins
    'target_reduction_start_adx': 20.0,

    # ADX value where maximum reduction is applied
    'target_reduction_max_adx': 40.0,
}


# =============================================================================
# REGIME-SPECIFIC BEHAVIOR MATRIX
# =============================================================================

# How the system behaves in each regime for counter-trend trades
REGIME_BEHAVIOR = {
    'trending_normal': {
        'allow_counter_trend': False,  # Block counter-trend entirely
        'target_multiplier': 0.6,      # If allowed, use 60% of normal target
        'confidence_adjustment': -0.5   # Reduce confidence by 50%
    },
    'trending_high_vol': {
        'allow_counter_trend': False,  # Block counter-trend entirely
        'target_multiplier': 0.5,      # If allowed, use 50% of normal target
        'confidence_adjustment': -0.6   # Reduce confidence by 60%
    },
    'ranging_normal': {
        'allow_counter_trend': True,   # Counter-trend is good here
        'target_multiplier': 1.2,      # Use 120% of normal target
        'confidence_adjustment': 0.15   # Boost confidence by 15%
    },
    'ranging_low_vol': {
        'allow_counter_trend': True,   # Best scenario for counter-trend
        'target_multiplier': 1.3,      # Use 130% of normal target
        'confidence_adjustment': 0.20   # Boost confidence by 20%
    },
    'transitional': {
        'allow_counter_trend': True,   # Allow but be cautious
        'target_multiplier': 0.8,      # Slightly reduced target
        'confidence_adjustment': -0.15  # Slight confidence reduction
    },
    'high_vol_chaos': {
        'allow_counter_trend': True,   # Allow but very cautious
        'target_multiplier': 0.6,      # Very tight targets
        'confidence_adjustment': -0.4   # Significant confidence reduction
    }
}


# =============================================================================
# POSITION SIZING ADJUSTMENTS
# =============================================================================

POSITION_SIZING_CONFIG = {
    # Reduce position size for counter-trend trades
    # Applied as a multiplier on top of normal position sizing
    'counter_trend_size_multiplier': 0.75,  # Use 75% of normal size

    # Regime-specific position size multipliers (already in risk_management.py)
    'regime_multipliers': {
        'trending_normal': 1.0,
        'trending_high_vol': 0.8,
        'ranging_normal': 0.7,
        'ranging_low_vol': 0.5,
        'high_vol_chaos': 0.4,
        'transitional': 0.6,
    }
}


# =============================================================================
# TREND DETECTION PARAMETERS
# =============================================================================

TREND_DETECTION_CONFIG = {
    # EMA periods for trend direction detection
    'fast_ema_period': 20,
    'slow_ema_period': 50,

    # Lookback period for regime detection
    'regime_lookback': 100,

    # ADX period for trend strength calculation
    'adx_period': 14,
}


# =============================================================================
# LOGGING AND MONITORING
# =============================================================================

LOGGING_CONFIG = {
    # Log all counter-trend trade decisions
    'log_counter_trend_decisions': True,

    # Log filter applications
    'log_filter_applications': True,

    # Include detailed regime information in logs
    'verbose_regime_logging': True,
}


# =============================================================================
# BACKTESTING COMPARISON SCENARIOS
# =============================================================================

# Use these configurations to compare different strategies in backtesting
BACKTEST_SCENARIOS = {
    'aggressive': {
        'block_counter_trends_in_strong_trends': False,
        'counter_trend_confidence_penalty': 0.3,
        'ranging_confidence_boost': 0.20,
    },
    'conservative': {
        'block_counter_trends_in_strong_trends': True,
        'counter_trend_confidence_penalty': 0.6,
        'ranging_confidence_boost': 0.10,
    },
    'moderate': {
        'block_counter_trends_in_strong_trends': True,
        'counter_trend_confidence_penalty': 0.5,
        'ranging_confidence_boost': 0.15,
    },
    'disabled': {
        'enable_filtering': False,
    }
}


def get_config(scenario='moderate'):
    """
    Get configuration for a specific scenario

    Args:
        scenario: One of 'aggressive', 'conservative', 'moderate', 'disabled'

    Returns:
        Configuration dictionary
    """
    if scenario not in BACKTEST_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Choose from {list(BACKTEST_SCENARIOS.keys())}")

    # Start with default config
    config = COUNTER_TREND_FILTER_CONFIG.copy()

    # Apply scenario-specific overrides
    config.update(BACKTEST_SCENARIOS[scenario])

    return config


if __name__ == '__main__':
    print("Counter-Trend Trading Configuration")
    print("=" * 70)
    print("\nDefault Configuration:")
    for key, value in COUNTER_TREND_FILTER_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n\nRegime Behavior Matrix:")
    for regime, behavior in REGIME_BEHAVIOR.items():
        print(f"\n{regime}:")
        for key, value in behavior.items():
            print(f"  {key}: {value}")

    print("\n\nBacktest Scenarios:")
    for scenario_name in BACKTEST_SCENARIOS.keys():
        print(f"  - {scenario_name}")
        config = get_config(scenario_name)
        print(f"    Block counter-trends: {config.get('block_counter_trends_in_strong_trends', 'N/A')}")
        print(f"    Enabled: {config.get('enable_filtering', True)}")
