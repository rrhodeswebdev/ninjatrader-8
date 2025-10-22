"""
Signal Quality Optimizer

Advanced techniques to reduce signal frequency and improve signal quality through:
1. Multi-factor confidence scoring with proper calibration
2. Kelly Criterion-based position sizing that naturally filters low-quality signals
3. Information Coefficient (IC) scoring for predictive value
4. Sharpe-optimal signal thresholding
5. Precision-recall optimization for optimal trade-off

Based on quantitative finance principles from:
- Advances in Financial Machine Learning (Marcos López de Prado)
- Evidence-Based Technical Analysis (David Aronson)
- Quantitative Trading (Ernest Chan)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats
from sklearn.metrics import precision_score, recall_score, roc_curve, auc


class SignalQualityScorer:
    """
    Advanced signal quality scoring that goes beyond simple probability

    Combines multiple quantitative measures:
    1. Model probability (base confidence)
    2. Directional conviction (separation between long/short)
    3. Trend alignment (Hurst exponent, momentum)
    4. Volatility regime appropriateness
    5. Information coefficient (if historical performance available)
    """

    def __init__(
        self,
        min_directional_edge: float = 0.15,  # Minimum prob difference between long/short
        min_trend_strength: float = 0.55,    # Minimum Hurst for trend boost
        counter_trend_penalty: float = 0.70,  # Multiply confidence by this for counter-trend
        volatility_scale_factor: float = 2.0  # Scale confidence by (current_vol / avg_vol)
    ):
        self.min_directional_edge = min_directional_edge
        self.min_trend_strength = min_trend_strength
        self.counter_trend_penalty = counter_trend_penalty
        self.volatility_scale_factor = volatility_scale_factor

        # Track historical performance for IC calculation
        self.prediction_history = []
        self.outcome_history = []

    def calculate_multi_factor_score(
        self,
        prob_short: float,
        prob_hold: float,
        prob_long: float,
        recent_bars_df: pd.DataFrame,
        signal: str
    ) -> Dict[str, float]:
        """
        Calculate advanced multi-factor confidence score

        Returns a dictionary with:
        - raw_confidence: Original model probability
        - quality_score: Adjusted score based on multiple factors (0-100)
        - directional_edge: Separation between long/short probabilities
        - trend_alignment: How well signal aligns with prevailing trend
        - volatility_factor: Adjustment for current volatility regime
        - final_confidence: Combined confidence score (0-1)
        """

        # 1. BASE CONFIDENCE (Model Probability)
        if signal == 'long':
            raw_confidence = prob_long
        elif signal == 'short':
            raw_confidence = prob_short
        else:  # hold
            raw_confidence = prob_hold

        # 2. DIRECTIONAL EDGE (Conviction)
        # How much stronger is the winning direction vs the other?
        directional_edge = abs(prob_long - prob_short)

        # Edge quality scoring (0-1 scale)
        if directional_edge >= 0.30:
            edge_score = 1.0  # Extremely strong
        elif directional_edge >= 0.20:
            edge_score = 0.8  # Very strong
        elif directional_edge >= 0.15:
            edge_score = 0.6  # Strong
        elif directional_edge >= 0.10:
            edge_score = 0.4  # Moderate
        else:
            edge_score = 0.2  # Weak

        # 3. TREND ALIGNMENT
        trend_score = 1.0
        trend_alignment = 0.0

        if len(recent_bars_df) >= 100:
            from model import calculate_hurst_exponent

            # Calculate Hurst exponent (>0.5 = trending, <0.5 = mean reverting)
            hurst, _ = calculate_hurst_exponent(recent_bars_df['close'].tail(100).values)

            # Calculate trend slope
            recent_closes = recent_bars_df['close'].tail(20).values
            trend_slope = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]

            # Determine if we're in a trend
            is_trending = hurst > self.min_trend_strength
            is_uptrend = is_trending and trend_slope > 0.001
            is_downtrend = is_trending and trend_slope < -0.001

            # Calculate alignment score
            if signal == 'long':
                if is_uptrend:
                    trend_alignment = 1.0  # Perfect alignment
                    trend_score = 1.3  # Boost
                elif is_downtrend:
                    trend_alignment = -1.0  # Counter-trend
                    trend_score = self.counter_trend_penalty  # Penalty
                else:
                    trend_alignment = 0.0  # Neutral
            elif signal == 'short':
                if is_downtrend:
                    trend_alignment = 1.0  # Perfect alignment
                    trend_score = 1.3  # Boost
                elif is_uptrend:
                    trend_alignment = -1.0  # Counter-trend
                    trend_score = self.counter_trend_penalty  # Penalty
                else:
                    trend_alignment = 0.0  # Neutral

        # 4. VOLATILITY REGIME ADJUSTMENT
        # In high volatility, require higher confidence
        # In low volatility, signals may be more reliable
        volatility_factor = 1.0

        if len(recent_bars_df) >= 50:
            returns = recent_bars_df['close'].pct_change().dropna()
            current_vol = returns.tail(20).std()  # Recent volatility
            avg_vol = returns.std()  # Historical average

            # Volatility ratio: >1 means higher than average
            vol_ratio = current_vol / (avg_vol + 1e-8)

            # Penalize signals in high volatility (less predictable)
            # Reward signals in low volatility (more predictable)
            if vol_ratio > 1.5:  # High volatility
                volatility_factor = 0.8
            elif vol_ratio > 1.2:  # Elevated volatility
                volatility_factor = 0.9
            elif vol_ratio < 0.7:  # Low volatility
                volatility_factor = 1.1
            elif vol_ratio < 0.5:  # Very low volatility
                volatility_factor = 1.2

        # 5. COMBINED QUALITY SCORE (0-100 scale)
        # Weight the different factors
        quality_score = (
            raw_confidence * 40 +        # 40% from model
            edge_score * 30 +             # 30% from directional conviction
            (trend_alignment + 1) * 15 +  # 15% from trend alignment (scaled 0-2 -> 0-30)
            volatility_factor * 15        # 15% from volatility regime
        )

        # 6. FINAL CONFIDENCE (0-1 scale)
        # Apply all adjustments to raw confidence
        final_confidence = raw_confidence * edge_score * trend_score * volatility_factor
        final_confidence = np.clip(final_confidence, 0.0, 1.0)

        return {
            'raw_confidence': raw_confidence,
            'quality_score': quality_score,
            'directional_edge': directional_edge,
            'edge_score': edge_score,
            'trend_alignment': trend_alignment,
            'trend_score': trend_score,
            'volatility_factor': volatility_factor,
            'final_confidence': final_confidence
        }

    def calculate_information_coefficient(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        method: str = 'spearman'
    ) -> float:
        """
        Calculate Information Coefficient (IC) - correlation between predictions and outcomes

        IC measures the predictive value of the model:
        - IC > 0.05: Decent predictive power
        - IC > 0.10: Good predictive power
        - IC > 0.15: Excellent predictive power

        Args:
            predictions: Array of model predictions (e.g., confidence scores)
            outcomes: Array of actual outcomes (e.g., forward returns)
            method: 'spearman' (default, more robust) or 'pearson'

        Returns:
            Information Coefficient (-1 to 1)
        """
        if len(predictions) < 20:
            return 0.0  # Not enough data

        if method == 'spearman':
            ic, p_value = stats.spearmanr(predictions, outcomes)
        else:
            ic, p_value = stats.pearsonr(predictions, outcomes)

        # Return 0 if not statistically significant (p > 0.05)
        if p_value > 0.05:
            return 0.0

        return ic

    def add_prediction_outcome(self, prediction: float, outcome: float):
        """
        Track a prediction and its outcome for IC calculation

        Args:
            prediction: Model confidence or score
            outcome: Actual result (e.g., trade P&L, forward return)
        """
        self.prediction_history.append(prediction)
        self.outcome_history.append(outcome)

        # Keep only last 500 predictions to avoid memory bloat
        if len(self.prediction_history) > 500:
            self.prediction_history = self.prediction_history[-500:]
            self.outcome_history = self.outcome_history[-500:]

    def get_current_ic(self) -> Optional[float]:
        """Get current Information Coefficient from recent predictions"""
        if len(self.prediction_history) < 20:
            return None

        return self.calculate_information_coefficient(
            np.array(self.prediction_history),
            np.array(self.outcome_history)
        )


class AdaptiveThresholdOptimizer:
    """
    Dynamically optimize confidence threshold to maximize Sharpe ratio or win rate

    Uses historical performance to find the optimal threshold that:
    1. Maximizes risk-adjusted returns (Sharpe ratio)
    2. Maintains acceptable win rate
    3. Balances precision vs recall
    """

    def __init__(
        self,
        target_sharpe: float = 2.0,
        min_win_rate: float = 0.55,
        min_trades_per_day: int = 2,
        lookback_trades: int = 100
    ):
        self.target_sharpe = target_sharpe
        self.min_win_rate = min_win_rate
        self.min_trades_per_day = min_trades_per_day
        self.lookback_trades = lookback_trades

        # Historical data
        self.trade_history = []  # List of {confidence, pnl, signal} dicts

    def add_trade(self, confidence: float, pnl: float, signal: str):
        """Record a completed trade"""
        self.trade_history.append({
            'confidence': confidence,
            'pnl': pnl,
            'signal': signal,
            'win': pnl > 0
        })

        # Keep only recent trades
        if len(self.trade_history) > self.lookback_trades * 2:
            self.trade_history = self.trade_history[-self.lookback_trades:]

    def optimize_threshold(
        self,
        min_threshold: float = 0.20,
        max_threshold: float = 0.90,
        step: float = 0.05
    ) -> Dict[str, float]:
        """
        Find optimal confidence threshold by testing different values

        Returns:
            Dictionary with optimal_threshold, expected_sharpe, win_rate, trades_per_day
        """
        if len(self.trade_history) < 30:
            return {
                'optimal_threshold': 0.25,  # Default
                'expected_sharpe': 0.0,
                'win_rate': 0.0,
                'trades_per_day': 0.0,
                'status': 'insufficient_data'
            }

        df = pd.DataFrame(self.trade_history)

        best_threshold = min_threshold
        best_sharpe = -999

        results = []

        # Test different threshold values
        for threshold in np.arange(min_threshold, max_threshold, step):
            # Filter trades above this threshold
            filtered_trades = df[df['confidence'] >= threshold]

            if len(filtered_trades) < 10:
                continue  # Not enough trades

            # Calculate metrics
            win_rate = filtered_trades['win'].mean()
            avg_pnl = filtered_trades['pnl'].mean()
            std_pnl = filtered_trades['pnl'].std()

            # Sharpe ratio (annualized assuming ~250 trading days)
            if std_pnl > 0:
                sharpe = (avg_pnl / std_pnl) * np.sqrt(250)
            else:
                sharpe = 0.0

            # Estimate trades per day (assuming data spans multiple days)
            num_trades = len(filtered_trades)

            results.append({
                'threshold': threshold,
                'sharpe': sharpe,
                'win_rate': win_rate,
                'num_trades': num_trades,
                'avg_pnl': avg_pnl
            })

            # Update best threshold based on Sharpe and constraints
            if sharpe > best_sharpe and win_rate >= self.min_win_rate:
                best_sharpe = sharpe
                best_threshold = threshold

        # Find the result for best threshold
        best_result = [r for r in results if r['threshold'] == best_threshold][0]

        return {
            'optimal_threshold': best_threshold,
            'expected_sharpe': best_sharpe,
            'win_rate': best_result['win_rate'],
            'avg_pnl': best_result['avg_pnl'],
            'num_trades': best_result['num_trades'],
            'status': 'optimized',
            'all_results': results  # For analysis
        }

    def get_precision_recall_curve(self) -> Dict:
        """
        Calculate precision-recall tradeoff at different thresholds

        Useful for understanding the tradeoff between:
        - Precision: When you trade, how often do you win?
        - Recall: Of all profitable opportunities, how many did you capture?
        """
        if len(self.trade_history) < 30:
            return {'error': 'insufficient_data'}

        df = pd.DataFrame(self.trade_history)

        thresholds = np.arange(0.1, 0.95, 0.05)
        precision_values = []
        recall_values = []

        # Total number of winning trades available
        total_wins = df['win'].sum()

        for threshold in thresholds:
            filtered = df[df['confidence'] >= threshold]

            if len(filtered) == 0:
                precision_values.append(0)
                recall_values.append(0)
                continue

            # Precision: Of trades taken, what % won?
            precision = filtered['win'].mean()

            # Recall: Of all winning trades, what % did we capture?
            recall = filtered['win'].sum() / total_wins if total_wins > 0 else 0

            precision_values.append(precision)
            recall_values.append(recall)

        return {
            'thresholds': thresholds.tolist(),
            'precision': precision_values,
            'recall': recall_values
        }


class KellyBasedFilter:
    """
    Use Kelly Criterion to naturally filter low-quality signals

    The Kelly Criterion calculates optimal position size based on edge and odds.
    If Kelly < some minimum, the signal isn't worth trading.

    This is more principled than arbitrary confidence thresholds.
    """

    def __init__(
        self,
        min_kelly_fraction: float = 0.02,  # Minimum 2% Kelly to trade
        max_kelly_fraction: float = 0.25,  # Cap at 25% Kelly (quarter-Kelly)
        default_win_rate: float = 0.55,
        default_avg_win: float = 1.0,
        default_avg_loss: float = 1.0
    ):
        self.min_kelly_fraction = min_kelly_fraction
        self.max_kelly_fraction = max_kelly_fraction

        # Historical statistics (updated from actual trades)
        self.win_rate = default_win_rate
        self.avg_win = default_avg_win
        self.avg_loss = default_avg_loss

        self.trade_history = []

    def calculate_kelly_size(
        self,
        confidence: float,
        expected_win: float = None,
        expected_loss: float = None
    ) -> float:
        """
        Calculate Kelly fraction for a given signal

        Kelly Formula: f* = (p*b - q) / b
        where:
            p = win probability (can use confidence as proxy)
            q = 1 - p
            b = avg_win / avg_loss (win/loss ratio)

        Returns:
            Kelly fraction (0-1, or 0 if signal doesn't meet minimum)
        """
        # Use confidence as win probability estimate
        p = confidence
        q = 1 - p

        # Use provided or historical win/loss sizes
        win_size = expected_win if expected_win is not None else self.avg_win
        loss_size = expected_loss if expected_loss is not None else self.avg_loss

        if loss_size <= 0:
            return 0.0

        b = win_size / loss_size

        # Full Kelly
        kelly = (p * b - q) / b

        # Apply maximum Kelly limit (quarter-Kelly is common)
        kelly = np.clip(kelly, 0.0, self.max_kelly_fraction)

        # Filter out if below minimum
        if kelly < self.min_kelly_fraction:
            return 0.0

        return kelly

    def should_trade(self, confidence: float) -> Tuple[bool, float, str]:
        """
        Determine if signal is worth trading based on Kelly Criterion

        Returns:
            (should_trade, kelly_fraction, reason)
        """
        kelly = self.calculate_kelly_size(confidence)

        if kelly >= self.min_kelly_fraction:
            return True, kelly, f"Kelly={kelly:.3f} (edge detected)"
        else:
            return False, kelly, f"Kelly={kelly:.3f} < min {self.min_kelly_fraction:.3f} (no edge)"

    def update_statistics(self, pnl: float):
        """Update historical win/loss statistics from completed trade"""
        self.trade_history.append(pnl)

        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]

        if len(self.trade_history) >= 20:
            wins = [x for x in self.trade_history if x > 0]
            losses = [abs(x) for x in self.trade_history if x < 0]

            if len(wins) > 0:
                self.win_rate = len(wins) / len(self.trade_history)
                self.avg_win = np.mean(wins)

            if len(losses) > 0:
                self.avg_loss = np.mean(losses)


# Example usage and testing
if __name__ == '__main__':
    print("="*70)
    print("SIGNAL QUALITY OPTIMIZER - TESTING")
    print("="*70)

    # Test 1: Multi-factor scoring
    print("\n1. Testing Multi-Factor Confidence Scoring")
    print("-" * 70)

    scorer = SignalQualityScorer()

    # Simulate some predictions
    test_cases = [
        {
            'prob_short': 0.20,
            'prob_hold': 0.30,
            'prob_long': 0.50,
            'signal': 'long',
            'description': 'Moderate long with weak edge'
        },
        {
            'prob_short': 0.15,
            'prob_hold': 0.25,
            'prob_long': 0.60,
            'signal': 'long',
            'description': 'Strong long with good edge'
        },
        {
            'prob_short': 0.40,
            'prob_hold': 0.35,
            'prob_long': 0.25,
            'signal': 'short',
            'description': 'Weak short with poor edge'
        },
    ]

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

    for case in test_cases:
        result = scorer.calculate_multi_factor_score(
            case['prob_short'],
            case['prob_hold'],
            case['prob_long'],
            df_sample,
            case['signal']
        )

        print(f"\n{case['description']}:")
        print(f"  Raw Confidence: {result['raw_confidence']:.3f}")
        print(f"  Quality Score: {result['quality_score']:.1f}/100")
        print(f"  Directional Edge: {result['directional_edge']:.3f}")
        print(f"  Final Confidence: {result['final_confidence']:.3f}")

    # Test 2: Kelly-based filtering
    print("\n\n2. Testing Kelly-Based Signal Filtering")
    print("-" * 70)

    kelly_filter = KellyBasedFilter(min_kelly_fraction=0.02)

    confidence_levels = [0.30, 0.45, 0.55, 0.65, 0.75, 0.85]

    for conf in confidence_levels:
        should_trade, kelly, reason = kelly_filter.should_trade(conf)
        status = "✓ TRADE" if should_trade else "✗ SKIP"
        print(f"Confidence {conf:.2f}: {status} - {reason}")

    # Test 3: Information Coefficient
    print("\n\n3. Testing Information Coefficient")
    print("-" * 70)

    # Simulate predictions and outcomes
    np.random.seed(42)
    n_trades = 100

    # Good model: predictions correlate with outcomes
    predictions_good = np.random.rand(n_trades)
    outcomes_good = predictions_good + np.random.randn(n_trades) * 0.3

    # Bad model: no correlation
    predictions_bad = np.random.rand(n_trades)
    outcomes_bad = np.random.randn(n_trades)

    ic_good = scorer.calculate_information_coefficient(predictions_good, outcomes_good)
    ic_bad = scorer.calculate_information_coefficient(predictions_bad, outcomes_bad)

    print(f"Good Model IC: {ic_good:.4f} (Strong predictive power)")
    print(f"Bad Model IC: {ic_bad:.4f} (No predictive power)")

    print("\n" + "="*70)
    print("Testing complete. See implementation above for integration.")
    print("="*70)
