"""
Market Regime Detection Module

Identifies distinct market conditions (trending, mean-reverting, volatile, quiet)
to adapt trading strategy and improve signal quality.
"""

import numpy as np
from typing import Tuple, Dict, List
from scipy import stats


class RegimeDetector:
    """
    Detect and classify market regimes using statistical properties

    Regimes:
    - trending: Persistent directional movement (Hurst > 0.55)
    - mean_reverting: Price reverts to mean (negative autocorrelation)
    - high_vol: Elevated volatility (>75th percentile)
    - low_vol: Suppressed volatility (<25th percentile)
    """

    def __init__(self, lookback: int = 100, volatility_window: int = 20):
        """
        Args:
            lookback: Bars for regime analysis
            volatility_window: Window for volatility calculation
        """
        self.lookback = lookback
        self.volatility_window = volatility_window
        self._historical_vol: List[float] = []

    def detect_regime(self, price_series: np.ndarray) -> Tuple[str, float]:
        """
        Detect current market regime

        Args:
            price_series: Price history

        Returns:
            (regime_name, confidence_score)
            regime_name: 'trending', 'mean_reverting', 'high_vol', 'low_vol'
            confidence_score: 0-1, confidence in regime classification
        """
        if len(price_series) < self.lookback:
            return 'unknown', 0.0

        # Calculate returns
        returns = np.diff(price_series) / price_series[:-1]

        if len(returns) == 0:
            return 'unknown', 0.0

        # 1. Trend persistence: Hurst exponent
        hurst = self._calculate_hurst(returns[-self.lookback:])

        # 2. Mean reversion: Autocorrelation
        autocorr = self._calculate_autocorrelation(returns[-self.lookback:], lag=1)

        # 3. Volatility regime
        volatility = np.std(returns[-self.volatility_window:]) * np.sqrt(252 * 390)

        # Update historical volatility
        self._historical_vol.append(volatility)
        if len(self._historical_vol) > 1000:
            self._historical_vol.pop(0)

        # Calculate volatility percentile
        if len(self._historical_vol) > 20:
            vol_percentile = stats.percentileofscore(self._historical_vol, volatility)
        else:
            vol_percentile = 50.0

        # Regime classification with confidence
        regimes_scores = {}

        # Trending regime (persistent movement)
        if hurst > 0.55 and autocorr > -0.1:
            trend_confidence = min((hurst - 0.5) * 2, 1.0)
            regimes_scores['trending'] = trend_confidence

        # Mean reverting regime (oscillating)
        if autocorr < -0.15:
            mr_confidence = min(abs(autocorr) / 0.5, 1.0)
            regimes_scores['mean_reverting'] = mr_confidence

        # Volatility regimes
        if vol_percentile > 75:
            hv_confidence = (vol_percentile - 75) / 25
            regimes_scores['high_vol'] = hv_confidence
        elif vol_percentile < 25:
            lv_confidence = (25 - vol_percentile) / 25
            regimes_scores['low_vol'] = lv_confidence

        # Return regime with highest confidence
        if regimes_scores:
            best_regime = max(regimes_scores, key=regimes_scores.get)
            confidence = regimes_scores[best_regime]
            return best_regime, float(confidence)

        # Default to low confidence unknown
        return 'ranging', 0.3

    def _calculate_hurst(self, returns: np.ndarray, lags: int = 20) -> float:
        """
        Calculate Hurst exponent for trend persistence

        H > 0.5: Trending (persistent)
        H = 0.5: Random walk
        H < 0.5: Mean reverting (anti-persistent)

        Args:
            returns: Return series
            lags: Number of lags to analyze

        Returns:
            Hurst exponent (0-1)
        """
        if len(returns) < lags:
            return 0.5

        try:
            # Calculate range/std for different lag sizes
            lags_range = range(2, min(lags, len(returns) // 2))

            if len(list(lags_range)) < 2:
                return 0.5

            tau = []
            for lag in lags_range:
                # Split returns into chunks
                n_chunks = len(returns) // lag
                if n_chunks < 1:
                    continue

                chunks_std = []
                for i in range(n_chunks):
                    chunk = returns[i*lag:(i+1)*lag]
                    if len(chunk) > 1:
                        chunks_std.append(np.std(chunk))

                if chunks_std:
                    tau.append(np.mean(chunks_std))

            if len(tau) < 2:
                return 0.5

            # Fit log(tau) vs log(lag)
            log_lags = np.log(list(lags_range)[:len(tau)])
            log_tau = np.log(tau)

            # Remove any inf/nan
            valid_mask = np.isfinite(log_lags) & np.isfinite(log_tau)
            if np.sum(valid_mask) < 2:
                return 0.5

            poly = np.polyfit(log_lags[valid_mask], log_tau[valid_mask], 1)
            hurst = poly[0]

            # Hurst should be between 0 and 1
            hurst = np.clip(hurst, 0.0, 1.0)

            return float(hurst)
        except Exception:
            return 0.5

    def _calculate_autocorrelation(self, returns: np.ndarray, lag: int = 1) -> float:
        """
        Calculate autocorrelation at given lag

        Negative autocorrelation suggests mean reversion
        Positive autocorrelation suggests momentum/trending

        Args:
            returns: Return series
            lag: Lag for autocorrelation

        Returns:
            Autocorrelation coefficient (-1 to 1)
        """
        if len(returns) < lag + 1:
            return 0.0

        try:
            # Pearson correlation between returns[t] and returns[t-lag]
            returns_current = returns[lag:]
            returns_lagged = returns[:-lag]

            if len(returns_current) < 2:
                return 0.0

            corr_matrix = np.corrcoef(returns_current, returns_lagged)
            autocorr = corr_matrix[0, 1]

            if np.isnan(autocorr):
                return 0.0

            return float(autocorr)
        except Exception:
            return 0.0

    def get_regime_params(self, regime: str) -> Dict[str, float]:
        """
        Get recommended trading parameters for regime

        Args:
            regime: Regime name

        Returns:
            Dictionary of recommended parameters
        """
        params = {
            'trending': {
                'signal_threshold': 0.55,  # Lower threshold, easier to enter
                'stop_multiplier': 2.0,    # Wider stops for trends
                'target_multiplier': 4.0,  # Larger targets
                'trailing_stop': True,
                'position_size_mult': 1.2, # Larger positions
            },
            'mean_reverting': {
                'signal_threshold': 0.70,  # Higher threshold, selective
                'stop_multiplier': 1.5,    # Tighter stops
                'target_multiplier': 2.0,  # Smaller targets
                'trailing_stop': False,
                'position_size_mult': 0.8, # Smaller positions
            },
            'high_vol': {
                'signal_threshold': 0.65,  # Higher threshold in volatility
                'stop_multiplier': 3.0,    # Much wider stops
                'target_multiplier': 5.0,  # Larger targets
                'trailing_stop': True,
                'position_size_mult': 0.7, # Reduce size in volatility
            },
            'low_vol': {
                'signal_threshold': 0.55,  # Can be more aggressive
                'stop_multiplier': 1.5,    # Tighter stops
                'target_multiplier': 2.5,  # Moderate targets
                'trailing_stop': False,
                'position_size_mult': 1.0, # Normal size
            },
            'ranging': {
                'signal_threshold': 0.60,  # Default threshold
                'stop_multiplier': 2.0,
                'target_multiplier': 3.0,
                'trailing_stop': False,
                'position_size_mult': 1.0,
            },
            'unknown': {
                'signal_threshold': 0.65,  # Conservative in uncertainty
                'stop_multiplier': 2.0,
                'target_multiplier': 3.0,
                'trailing_stop': False,
                'position_size_mult': 0.5, # Very small size
            }
        }

        return params.get(regime, params['unknown'])

    def get_volatility_percentile(self) -> float:
        """
        Get current volatility percentile vs historical

        Returns:
            Percentile (0-100)
        """
        if len(self._historical_vol) < 2:
            return 50.0

        current_vol = self._historical_vol[-1]
        percentile = stats.percentileofscore(self._historical_vol, current_vol)

        return float(percentile)


class AdaptiveRegimeFilter:
    """
    Filter trading signals based on regime suitability

    Some strategies work better in certain regimes.
    This filter prevents trading in unfavorable conditions.
    """

    def __init__(self, allowed_regimes: List[str] = None,
                 min_confidence: float = 0.5):
        """
        Args:
            allowed_regimes: List of regimes to trade (None = all)
            min_confidence: Minimum regime confidence to trade
        """
        self.allowed_regimes = allowed_regimes or ['trending', 'ranging', 'low_vol']
        self.min_confidence = min_confidence

    def should_trade(self, regime: str, confidence: float) -> bool:
        """
        Determine if trading is allowed in current regime

        Args:
            regime: Current regime
            confidence: Regime confidence

        Returns:
            True if should trade, False if should skip
        """
        if confidence < self.min_confidence:
            return False

        if regime not in self.allowed_regimes:
            return False

        return True

    def adjust_signal_strength(self, signal_strength: float,
                              regime: str, confidence: float) -> float:
        """
        Adjust signal strength based on regime favorability

        Args:
            signal_strength: Original signal strength (0-1)
            regime: Current regime
            confidence: Regime confidence

        Returns:
            Adjusted signal strength
        """
        # Boost signals in favorable regimes
        favorable_regimes = ['trending', 'low_vol']
        unfavorable_regimes = ['high_vol', 'mean_reverting']

        if regime in favorable_regimes:
            multiplier = 1.0 + (confidence * 0.2)  # Up to 20% boost
        elif regime in unfavorable_regimes:
            multiplier = 1.0 - (confidence * 0.3)  # Up to 30% reduction
        else:
            multiplier = 1.0

        adjusted = signal_strength * multiplier

        return float(np.clip(adjusted, 0.0, 1.0))
