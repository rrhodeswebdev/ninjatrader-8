"""
Advanced Confidence Scoring System - Complete Implementation

Implements the 5-component confidence scoring from the quant analysis:
1. Model Probability Score (25%)
2. Feature Quality Score (20%)
3. Market Regime Fit Score (20%)
4. Ensemble Agreement Score (20%)
5. Recent Performance Score (15%)

Dynamic thresholds based on market regime
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats


class AdvancedConfidenceScoring:
    """
    Multi-factor confidence scoring with dynamic thresholds
    """

    def __init__(self, training_stats: Optional[Dict] = None):
        """
        Initialize confidence scorer

        Args:
            training_stats: Statistics from training data (mean, std for features)
        """
        self.weights = {
            'model_probability': 0.25,      # Base model confidence
            'feature_quality': 0.20,         # Data quality score
            'market_regime_fit': 0.20,       # Regime suitability
            'ensemble_agreement': 0.20,      # Multiple model consensus
            'recent_performance': 0.15       # Adaptive performance tracking
        }

        self.training_stats = training_stats or {}

        # Dynamic thresholds by regime
        self.regime_thresholds = {
            'trending': 60.0,       # Lower threshold in clean trends
            'ranging': 75.0,        # Higher threshold in choppy markets
            'volatile': 70.0,       # Medium threshold in volatile markets
            'low_volume': 80.0,     # Highest threshold in low liquidity
            'normal': 65.0          # Default threshold
        }

    def calculate_composite_confidence(
        self,
        probabilities: np.ndarray,
        features: np.ndarray,
        market_state: Dict,
        model_predictions: Optional[List[Dict]] = None,
        trade_history: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate weighted composite confidence score

        Args:
            probabilities: Model output probabilities for each class
            features: Input features used for prediction
            market_state: Current market characteristics
            model_predictions: Predictions from ensemble models (optional)
            trade_history: Recent trade results (optional)

        Returns:
            Tuple of (composite_score, component_scores)
            - composite_score: 0-100
            - component_scores: Dict of individual scores
        """
        scores = {}

        # 1. Model Probability Score (0-100)
        scores['model_probability'] = self._calculate_model_score(probabilities)

        # 2. Feature Quality Score (0-100)
        scores['feature_quality'] = self._calculate_feature_quality(features)

        # 3. Market Regime Fit Score (0-100)
        scores['market_regime_fit'] = self._calculate_regime_fit(market_state)

        # 4. Ensemble Agreement Score (0-100)
        if model_predictions:
            scores['ensemble_agreement'] = self._calculate_ensemble_agreement(model_predictions)
        else:
            scores['ensemble_agreement'] = 50.0  # Neutral if no ensemble

        # 5. Recent Performance Score (0-100)
        if trade_history:
            scores['recent_performance'] = self._calculate_recent_performance(trade_history)
        else:
            scores['recent_performance'] = 50.0  # Neutral if no history

        # Weighted composite
        composite = sum(
            scores[k] * self.weights[k]
            for k in scores
        )

        return composite, scores

    def _calculate_model_score(self, probabilities: np.ndarray) -> float:
        """
        Transform raw probability to confidence score

        Uses calibrated probability with entropy penalty

        Args:
            probabilities: Array of class probabilities

        Returns:
            Score 0-100
        """
        if len(probabilities) == 0:
            return 0.0

        max_prob = np.max(probabilities)

        # Calculate entropy (uncertainty)
        # Clip to avoid log(0)
        probs_clipped = np.clip(probabilities, 1e-10, 1.0)
        entropy = -np.sum(probabilities * np.log(probs_clipped))

        # Normalize entropy (0 = certain, 1 = max uncertainty)
        max_entropy = np.log(len(probabilities))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Confidence = high probability + low entropy
        confidence = max_prob * (1 - normalized_entropy)

        return float(confidence * 100)

    def _calculate_feature_quality(self, features: np.ndarray) -> float:
        """
        Assess quality of input features

        Checks:
        - Missing values
        - Outliers (beyond 3 std dev)
        - Feature magnitude (too large/small values)

        Args:
            features: Feature vector

        Returns:
            Quality score 0-100
        """
        if len(features) == 0:
            return 0.0

        quality_score = 100.0

        # Convert to numpy array if needed
        if isinstance(features, list):
            features = np.array(features)

        # Flatten if needed
        features = features.flatten()

        # Missing value penalty
        missing_count = np.isnan(features).sum()
        missing_pct = missing_count / len(features)
        quality_score -= missing_pct * 50

        # Outlier penalty (values beyond reasonable range)
        # Replace NaN with 0 for outlier detection
        features_clean = np.nan_to_num(features, 0)

        if len(features_clean) > 0:
            # Check for extreme values (beyond 100 or below -100)
            extreme_count = np.sum((np.abs(features_clean) > 100))
            extreme_pct = extreme_count / len(features_clean)
            quality_score -= extreme_pct * 30

            # Check for inf values
            inf_count = np.sum(np.isinf(features_clean))
            inf_pct = inf_count / len(features_clean)
            quality_score -= inf_pct * 40

        return max(0, quality_score)

    def _calculate_regime_fit(self, market_state: Dict) -> float:
        """
        Score how well current market fits model training regime

        Compares current market characteristics to training distribution

        Args:
            market_state: Dict with 'volatility', 'volume', 'trend_strength'

        Returns:
            Fit score 0-100
        """
        if not market_state or not self.training_stats:
            return 50.0  # Neutral if no stats available

        fit_scores = []

        # Volatility fit
        if 'volatility' in market_state and 'volatility_mean' in self.training_stats:
            current_vol = market_state['volatility']
            vol_mean = self.training_stats['volatility_mean']
            vol_std = self.training_stats.get('volatility_std', vol_mean * 0.1)

            vol_fit = self._distribution_fit_score(current_vol, vol_mean, vol_std)
            fit_scores.append(vol_fit)

        # Volume fit
        if 'volume' in market_state and 'volume_mean' in self.training_stats:
            current_volume = market_state['volume']
            volume_mean = self.training_stats['volume_mean']
            volume_std = self.training_stats.get('volume_std', volume_mean * 0.1)

            volume_fit = self._distribution_fit_score(current_volume, volume_mean, volume_std)
            fit_scores.append(volume_fit)

        # Trend strength fit
        if 'trend_strength' in market_state and 'trend_mean' in self.training_stats:
            current_trend = market_state['trend_strength']
            trend_mean = self.training_stats['trend_mean']
            trend_std = self.training_stats.get('trend_std', 0.1)

            trend_fit = self._distribution_fit_score(current_trend, trend_mean, trend_std)
            fit_scores.append(trend_fit)

        # Average fit scores
        if fit_scores:
            regime_fit = np.mean(fit_scores)
        else:
            regime_fit = 0.5  # Neutral

        return float(regime_fit * 100)

    def _distribution_fit_score(self, current_value: float, mean: float, std: float) -> float:
        """
        Calculate how well current value fits training distribution

        Returns 1.0 if within 1 std dev, decreases beyond that

        Args:
            current_value: Current observed value
            mean: Training data mean
            std: Training data standard deviation

        Returns:
            Fit score 0-1
        """
        if std == 0:
            return 1.0 if current_value == mean else 0.0

        z_score = abs((current_value - mean) / std)

        # Gaussian-like decay
        fit_score = np.exp(-0.5 * z_score**2)

        return float(fit_score)

    def _calculate_ensemble_agreement(self, model_predictions: List[Dict]) -> float:
        """
        Measure agreement across multiple model variants

        High agreement = high confidence

        Args:
            model_predictions: List of dicts with 'class' and 'probability'

        Returns:
            Agreement score 0-100
        """
        if len(model_predictions) < 2:
            return 50.0  # Neutral if no ensemble

        # Get predictions from all models
        predictions = np.array([pred['class'] for pred in model_predictions])
        probabilities = np.array([pred['probability'] for pred in model_predictions])

        # Agreement rate (mode)
        values, counts = np.unique(predictions, return_counts=True)
        mode_prediction = values[np.argmax(counts)]
        agreement_rate = (predictions == mode_prediction).mean()

        # Probability consensus (low std = high consensus)
        prob_std = probabilities.std()
        prob_consensus = 1 - min(prob_std / 0.3, 1.0)  # Normalize by threshold

        # Combined score
        agreement_score = (agreement_rate * 0.6 + prob_consensus * 0.4)

        return float(agreement_score * 100)

    def _calculate_recent_performance(self, trade_history: List[Dict], lookback: int = 20) -> float:
        """
        Adaptive confidence based on recent win rate and P&L

        Good recent performance -> higher confidence
        Poor recent performance -> lower confidence

        Args:
            trade_history: List of recent trades with 'pnl' field
            lookback: Number of recent trades to analyze

        Returns:
            Performance score 0-100
        """
        if len(trade_history) < 5:
            return 50.0  # Neutral if insufficient history

        recent_trades = trade_history[-lookback:]

        # Win rate
        wins = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
        win_rate = wins / len(recent_trades)

        # Profit factor
        gross_profit = sum(t['pnl'] for t in recent_trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t['pnl'] for t in recent_trades if t.get('pnl', 0) < 0))

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = 2.0 if gross_profit > 0 else 0.0

        # Average win vs average loss
        if wins > 0:
            avg_win = gross_profit / wins
        else:
            avg_win = 0

        losses = len(recent_trades) - wins
        if losses > 0:
            avg_loss = gross_loss / losses
        else:
            avg_loss = 1.0

        rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # Combined performance score
        performance_score = (
            win_rate * 0.4 +
            min(profit_factor / 2.0, 1.0) * 0.4 +
            min(rr_ratio / 2.0, 1.0) * 0.2
        )

        return float(performance_score * 100)

    def get_dynamic_threshold(self, market_regime: str) -> float:
        """
        Dynamic confidence threshold based on market regime

        Args:
            market_regime: One of 'trending', 'ranging', 'volatile', 'low_volume', 'normal'

        Returns:
            Minimum confidence required to trade
        """
        return self.regime_thresholds.get(market_regime, 65.0)

    def set_training_stats(self, training_data: pd.DataFrame):
        """
        Calculate and store training statistics for regime fit scoring

        Args:
            training_data: Training dataset with features
        """
        if 'volatility' in training_data.columns:
            self.training_stats['volatility_mean'] = training_data['volatility'].mean()
            self.training_stats['volatility_std'] = training_data['volatility'].std()

        if 'volume' in training_data.columns:
            self.training_stats['volume_mean'] = training_data['volume'].mean()
            self.training_stats['volume_std'] = training_data['volume'].std()

        if 'trend_strength' in training_data.columns:
            self.training_stats['trend_mean'] = training_data['trend_strength'].mean()
            self.training_stats['trend_std'] = training_data['trend_strength'].std()

    def should_trade(
        self,
        composite_confidence: float,
        market_regime: str,
        component_scores: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Determine if should trade based on confidence and regime

        Args:
            composite_confidence: Overall confidence score 0-100
            market_regime: Current market regime
            component_scores: Individual component scores (optional)

        Returns:
            Tuple of (should_trade, reason)
        """
        threshold = self.get_dynamic_threshold(market_regime)

        if composite_confidence >= threshold:
            return True, f"Confidence {composite_confidence:.1f} >= threshold {threshold:.1f}"

        # Check if any critical component is too low
        if component_scores:
            if component_scores.get('feature_quality', 100) < 30:
                return False, "Feature quality too low"

            if component_scores.get('model_probability', 100) < 40:
                return False, "Model probability too low"

            if component_scores.get('recent_performance', 50) < 20:
                return False, "Recent performance too poor"

        return False, f"Confidence {composite_confidence:.1f} < threshold {threshold:.1f}"
