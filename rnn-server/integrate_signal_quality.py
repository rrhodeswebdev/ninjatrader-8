"""
Integration Example: Signal Quality Improvements

This file shows how to integrate the signal quality optimizer into main.py

Choose your strategy:
1. Quick Win: Just raise threshold
2. Quality Score: Multi-factor scoring
3. Kelly Filter: Mathematical edge-based filtering
4. All Combined: Use all strategies together
"""

from signal_quality_optimizer import (
    SignalQualityScorer,
    KellyBasedFilter,
    AdaptiveThresholdOptimizer
)


# ============================================================================
# STRATEGY 1: QUICK WIN - Just raise the threshold
# ============================================================================

def quick_win_filter(confidence: float, min_threshold: float = 0.55) -> tuple:
    """
    Simplest approach: Just raise minimum confidence requirement

    Args:
        confidence: Model confidence (0-1)
        min_threshold: Minimum confidence to trade

    Returns:
        (should_trade, reason)
    """
    if confidence < min_threshold:
        return False, f"Confidence {confidence:.2%} below minimum {min_threshold:.2%}"
    return True, "Passed threshold"


# ============================================================================
# STRATEGY 2: QUALITY SCORE - Multi-factor evaluation
# ============================================================================

class QualityScoreFilter:
    """
    Use multi-factor quality scoring instead of simple threshold
    """

    def __init__(self, min_quality_score: float = 55):
        self.scorer = SignalQualityScorer(
            min_directional_edge=0.15,      # Require 15% edge
            min_trend_strength=0.55,         # Hurst > 0.55 for trend
            counter_trend_penalty=0.70,      # 30% penalty for counter-trend
            volatility_scale_factor=2.0
        )
        self.min_quality_score = min_quality_score

    def evaluate_signal(
        self,
        prob_short: float,
        prob_hold: float,
        prob_long: float,
        recent_bars_df,
        signal: str
    ) -> dict:
        """
        Evaluate signal quality

        Returns:
            {
                'should_trade': bool,
                'quality_score': float (0-100),
                'final_confidence': float (0-1),
                'reason': str,
                'details': dict
            }
        """
        # Calculate multi-factor score
        quality = self.scorer.calculate_multi_factor_score(
            prob_short, prob_hold, prob_long,
            recent_bars_df, signal
        )

        # Check if passes minimum quality
        should_trade = quality['quality_score'] >= self.min_quality_score

        # Determine position size scaling based on quality
        if quality['quality_score'] >= 75:
            size_multiplier = 1.0      # Excellent: Full size
        elif quality['quality_score'] >= 65:
            size_multiplier = 0.75     # Good: 75% size
        elif quality['quality_score'] >= 55:
            size_multiplier = 0.50     # Marginal: 50% size
        else:
            size_multiplier = 0.0      # Poor: Skip

        reason = f"Quality: {quality['quality_score']:.1f}/100"
        if quality['quality_score'] >= 75:
            reason += " (Excellent)"
        elif quality['quality_score'] >= 65:
            reason += " (Good)"
        elif quality['quality_score'] >= 55:
            reason += " (Marginal)"
        else:
            reason += " (Too low)"

        return {
            'should_trade': should_trade,
            'quality_score': quality['quality_score'],
            'final_confidence': quality['final_confidence'],
            'size_multiplier': size_multiplier,
            'reason': reason,
            'details': quality
        }


# ============================================================================
# STRATEGY 3: KELLY CRITERION - Mathematical edge filter
# ============================================================================

class KellySignalFilter:
    """
    Use Kelly Criterion to filter signals without mathematical edge
    """

    def __init__(self, min_kelly: float = 0.02):
        self.kelly_filter = KellyBasedFilter(
            min_kelly_fraction=min_kelly,   # Must have 2%+ Kelly edge
            max_kelly_fraction=0.25          # Cap at quarter-Kelly
        )

    def evaluate_signal(self, confidence: float) -> dict:
        """
        Check if signal has sufficient edge to trade

        Returns:
            {
                'should_trade': bool,
                'kelly_size': float,
                'position_multiplier': float (0-1),
                'reason': str
            }
        """
        should_trade, kelly_size, reason = self.kelly_filter.should_trade(confidence)

        # Scale position size by Kelly percentage
        # kelly_size is 0-0.25, we normalize to 0-1
        position_multiplier = min(1.0, kelly_size / 0.25) if kelly_size > 0 else 0.0

        return {
            'should_trade': should_trade,
            'kelly_size': kelly_size,
            'position_multiplier': position_multiplier,
            'reason': reason
        }

    def update_from_trade(self, pnl: float):
        """Update Kelly calculator with trade result"""
        self.kelly_filter.update_statistics(pnl)


# ============================================================================
# STRATEGY 4: COMBINED - Use all filters together
# ============================================================================

class CombinedSignalFilter:
    """
    Combine multiple filtering strategies for maximum quality
    """

    def __init__(
        self,
        min_confidence: float = 0.45,
        min_quality_score: float = 55,
        min_kelly: float = 0.02,
        use_quality_score: bool = True,
        use_kelly_filter: bool = True
    ):
        self.min_confidence = min_confidence
        self.use_quality_score = use_quality_score
        self.use_kelly_filter = use_kelly_filter

        # Initialize sub-filters
        if use_quality_score:
            self.quality_filter = QualityScoreFilter(min_quality_score)

        if use_kelly_filter:
            self.kelly_filter = KellySignalFilter(min_kelly)

        # Adaptive threshold optimizer
        self.threshold_optimizer = AdaptiveThresholdOptimizer(
            target_sharpe=2.0,
            min_win_rate=0.55
        )

    def evaluate_signal(
        self,
        prob_short: float,
        prob_hold: float,
        prob_long: float,
        recent_bars_df,
        signal: str,
        raw_confidence: float
    ) -> dict:
        """
        Apply all filters to signal

        Returns complete evaluation with recommendations
        """
        result = {
            'should_trade': True,
            'filters_passed': [],
            'filters_failed': [],
            'confidence': raw_confidence,
            'position_multiplier': 1.0,
            'reasons': []
        }

        # Filter 1: Minimum confidence threshold
        if raw_confidence < self.min_confidence:
            result['should_trade'] = False
            result['filters_failed'].append('min_confidence')
            result['reasons'].append(
                f"Confidence {raw_confidence:.2%} < minimum {self.min_confidence:.2%}"
            )
            result['position_multiplier'] = 0.0
            return result

        result['filters_passed'].append('min_confidence')

        # Filter 2: Quality Score (if enabled)
        if self.use_quality_score:
            quality_eval = self.quality_filter.evaluate_signal(
                prob_short, prob_hold, prob_long,
                recent_bars_df, signal
            )

            if not quality_eval['should_trade']:
                result['should_trade'] = False
                result['filters_failed'].append('quality_score')
                result['reasons'].append(quality_eval['reason'])
                result['position_multiplier'] = 0.0
                return result

            result['filters_passed'].append('quality_score')
            result['quality_score'] = quality_eval['quality_score']
            result['position_multiplier'] = min(
                result['position_multiplier'],
                quality_eval['size_multiplier']
            )
            result['reasons'].append(quality_eval['reason'])

        # Filter 3: Kelly Criterion (if enabled)
        if self.use_kelly_filter:
            kelly_eval = self.kelly_filter.evaluate_signal(raw_confidence)

            if not kelly_eval['should_trade']:
                result['should_trade'] = False
                result['filters_failed'].append('kelly_criterion')
                result['reasons'].append(kelly_eval['reason'])
                result['position_multiplier'] = 0.0
                return result

            result['filters_passed'].append('kelly_criterion')
            result['kelly_size'] = kelly_eval['kelly_size']
            result['position_multiplier'] = min(
                result['position_multiplier'],
                kelly_eval['position_multiplier']
            )
            result['reasons'].append(kelly_eval['reason'])

        # All filters passed
        result['reasons'].insert(0, " All filters passed")

        return result

    def update_from_trade(self, confidence: float, pnl: float, signal: str):
        """
        Update all adaptive components with trade result

        Args:
            confidence: Confidence of the trade
            pnl: Profit/loss of the trade
            signal: 'long' or 'short'
        """
        # Update threshold optimizer
        self.threshold_optimizer.add_trade(confidence, pnl, signal)

        # Update Kelly filter
        if self.use_kelly_filter:
            self.kelly_filter.update_from_trade(pnl)

    def optimize_threshold(self) -> dict:
        """
        Run threshold optimization based on recent trades

        Call this weekly or after every 50-100 trades
        """
        result = self.threshold_optimizer.optimize_threshold()

        if result['status'] == 'optimized':
            print("\n" + "="*70)
            print("THRESHOLD OPTIMIZATION RESULTS")
            print("="*70)
            print(f"Current threshold: {self.min_confidence:.2%}")
            print(f"Optimal threshold: {result['optimal_threshold']:.2%}")
            print(f"Expected Sharpe: {result['expected_sharpe']:.2f}")
            print(f"Expected win rate: {result['win_rate']:.2%}")
            print(f"Expected trades: {result['num_trades']}")

            # Update threshold
            self.min_confidence = result['optimal_threshold']
            print(f"\n Threshold updated to {self.min_confidence:.2%}")
        else:
            print(f"Threshold optimization: {result['status']}")

        return result


# ============================================================================
# EXAMPLE: How to integrate into main.py
# ============================================================================

def example_integration():
    """
    Example showing how to integrate into your main.py analysis endpoint
    """
    # Initialize filter (choose your strategy)

    # Option 1: Quick win (simple threshold)
    # Just change MIN_CONFIDENCE_THRESHOLD in main.py from 0.25 to 0.55

    # Option 2: Quality score filter
    filter_system = QualityScoreFilter(min_quality_score=55)

    # Option 3: Kelly filter
    # filter_system = KellySignalFilter(min_kelly=0.02)

    # Option 4: Combined (recommended)
    filter_system = CombinedSignalFilter(
        min_confidence=0.45,      # Base threshold
        min_quality_score=55,     # Quality threshold
        min_kelly=0.02,           # Kelly threshold
        use_quality_score=True,
        use_kelly_filter=True
    )

    # In your /analysis endpoint, replace the simple threshold check:
    """
    # OLD CODE (main.py ~line 380-386):
    signal = trade_params['signal']
    confidence = trade_params.get('confidence', 0.0)

    filtered_signal = signal
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        filtered_signal = "hold"
        print(f"Low confidence ({confidence*100:.2f}%) - Filtering {signal.upper()}  HOLD")

    # NEW CODE:
    signal = trade_params['signal']
    confidence = trade_params.get('confidence', 0.0)

    # Get model probabilities (you need to expose these from predict())
    prob_short, prob_hold, prob_long = 0.2, 0.3, 0.5  # Example

    # Evaluate signal quality
    evaluation = filter_system.evaluate_signal(
        prob_short, prob_hold, prob_long,
        current_data,  # Your recent bars DataFrame
        signal,
        confidence
    )

    # Apply filter decision
    if not evaluation['should_trade']:
        filtered_signal = "hold"
        print(f"Signal filtered: {', '.join(evaluation['reasons'])}")
    else:
        filtered_signal = signal
        print(f"Signal approved: {', '.join(evaluation['reasons'])}")

        # Optionally adjust position size based on quality
        base_contracts = trade_params.get('contracts', 1)
        adjusted_contracts = int(base_contracts * evaluation['position_multiplier'])
        trade_params['contracts'] = adjusted_contracts
        print(f"Position size: {adjusted_contracts} contracts (multiplier: {evaluation['position_multiplier']:.2f})")
    """

    print("See comments above for integration code")


# ============================================================================
# MAIN: Run examples
# ============================================================================

if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    print("="*70)
    print("SIGNAL QUALITY INTEGRATION EXAMPLES")
    print("="*70)

    # Create sample data
    dates = pd.date_range('2025-01-01', periods=150, freq='1min')
    prices = 4500 + np.cumsum(np.random.randn(150) * 2)
    df_sample = pd.DataFrame({
        'time': dates,
        'close': prices,
        'high': prices + np.random.rand(150) * 3,
        'low': prices - np.random.rand(150) * 3,
        'volume': np.random.randint(100, 1000, 150)
    })

    # Test signals
    test_signals = [
        {
            'prob_short': 0.25,
            'prob_hold': 0.35,
            'prob_long': 0.40,
            'signal': 'long',
            'confidence': 0.40,
            'description': 'Weak long signal'
        },
        {
            'prob_short': 0.15,
            'prob_hold': 0.25,
            'prob_long': 0.60,
            'signal': 'long',
            'confidence': 0.60,
            'description': 'Strong long signal'
        },
        {
            'prob_short': 0.70,
            'prob_hold': 0.15,
            'prob_long': 0.15,
            'signal': 'short',
            'confidence': 0.70,
            'description': 'Very strong short signal'
        },
    ]

    # Test each strategy
    print("\n" + "="*70)
    print("TESTING COMBINED FILTER")
    print("="*70)

    combined_filter = CombinedSignalFilter(
        min_confidence=0.45,
        min_quality_score=55,
        min_kelly=0.02
    )

    for test in test_signals:
        print(f"\n{test['description']}:")
        print(f"  Probabilities: SHORT={test['prob_short']:.2f}, HOLD={test['prob_hold']:.2f}, LONG={test['prob_long']:.2f}")

        result = combined_filter.evaluate_signal(
            test['prob_short'],
            test['prob_hold'],
            test['prob_long'],
            df_sample,
            test['signal'],
            test['confidence']
        )

        print(f"  Decision: {' TRADE' if result['should_trade'] else ' SKIP'}")
        print(f"  Filters passed: {', '.join(result['filters_passed'])}")
        if result['filters_failed']:
            print(f"  Filters failed: {', '.join(result['filters_failed'])}")
        print(f"  Position multiplier: {result['position_multiplier']:.2f}x")
        print(f"  Reasons: {'; '.join(result['reasons'])}")

    print("\n" + "="*70)
    print("See SIGNAL_QUALITY_IMPROVEMENT.md for full documentation")
    print("="*70)
