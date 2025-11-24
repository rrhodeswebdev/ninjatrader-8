"""
Probability Calibration Module

Calibrates raw neural network outputs to match empirical probabilities.
Ensures confidence thresholds are meaningful and reliable.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import pickle
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """
    Calibrate model probabilities using Isotonic Regression

    Raw neural network outputs are often poorly calibrated.
    A model might output 0.6 probability but actually be correct only 52% of the time.
    This calibrator learns the true mapping from raw outputs to empirical frequencies.
    """

    def __init__(self, out_of_bounds: str = 'clip'):
        """
        Args:
            out_of_bounds: How to handle out-of-bounds predictions ('clip' or 'nan')
        """
        self.calibrator = IsotonicRegression(out_of_bounds=out_of_bounds, y_min=0.0, y_max=1.0)
        self.is_fitted = False
        self._calibration_curve_data: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def fit(self, predictions: np.ndarray, actual_outcomes: np.ndarray) -> 'ProbabilityCalibrator':
        """
        Fit calibrator to empirical data

        Args:
            predictions: Raw model predictions (0-1)
            actual_outcomes: Actual binary outcomes (0 or 1)

        Returns:
            Self for chaining
        """
        predictions = np.asarray(predictions).flatten()
        actual_outcomes = np.asarray(actual_outcomes).flatten()

        if len(predictions) != len(actual_outcomes):
            raise ValueError("predictions and actual_outcomes must have same length")

        if len(predictions) < 10:
            logger.warning(f"Only {len(predictions)} samples for calibration - results may be unreliable")

        # Fit isotonic regression
        self.calibrator.fit(predictions, actual_outcomes)
        self.is_fitted = True

        # Store calibration curve for diagnostics
        self._calibration_curve_data = calibration_curve(
            actual_outcomes, predictions, n_bins=10, strategy='uniform'
        )

        logger.info(f"Calibrator fitted on {len(predictions)} samples")

        return self

    def calibrate(self, raw_probability: float) -> float:
        """
        Convert raw model output to calibrated probability

        Args:
            raw_probability: Raw model prediction (0-1)

        Returns:
            Calibrated probability matching empirical frequency
        """
        if not self.is_fitted:
            logger.warning("Calibrator not fitted - returning raw probability")
            return float(np.clip(raw_probability, 0.0, 1.0))

        calibrated = self.calibrator.predict([raw_probability])[0]
        return float(calibrated)

    def calibrate_batch(self, raw_probabilities: np.ndarray) -> np.ndarray:
        """
        Calibrate multiple predictions at once

        Args:
            raw_probabilities: Array of raw predictions

        Returns:
            Array of calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning("Calibrator not fitted - returning raw probabilities")
            return np.clip(raw_probabilities, 0.0, 1.0)

        calibrated = self.calibrator.predict(raw_probabilities)
        return calibrated

    def get_calibration_curve(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get calibration curve data for visualization

        Returns:
            (true_probabilities, predicted_probabilities) or None if not fitted
        """
        return self._calibration_curve_data

    def get_calibration_error(self) -> float:
        """
        Calculate Expected Calibration Error (ECE)

        ECE measures how well-calibrated the model is.
        Lower is better (0 = perfectly calibrated)

        Returns:
            Expected Calibration Error
        """
        if self._calibration_curve_data is None:
            return float('inf')

        true_probs, pred_probs = self._calibration_curve_data

        # ECE = average absolute difference between predicted and true probabilities
        ece = np.mean(np.abs(pred_probs - true_probs))

        return float(ece)

    def save(self, filepath: str):
        """
        Save calibrator to disk

        Args:
            filepath: Path to save calibrator
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'calibrator': self.calibrator,
                'is_fitted': self.is_fitted,
                'calibration_curve_data': self._calibration_curve_data
            }, f)

        logger.info(f"Calibrator saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ProbabilityCalibrator':
        """
        Load calibrator from disk

        Args:
            filepath: Path to load calibrator from

        Returns:
            Loaded ProbabilityCalibrator
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        calibrator = cls()
        calibrator.calibrator = data['calibrator']
        calibrator.is_fitted = data['is_fitted']
        calibrator._calibration_curve_data = data['calibration_curve_data']

        logger.info(f"Calibrator loaded from {filepath}")

        return calibrator


class ConfidenceIntervalEstimator:
    """
    Estimate confidence intervals for predictions

    Provides uncertainty quantification alongside point predictions
    """

    def __init__(self, n_bootstrap: int = 100):
        """
        Args:
            n_bootstrap: Number of bootstrap samples for interval estimation
        """
        self.n_bootstrap = n_bootstrap
        self.bootstrap_predictions: Optional[np.ndarray] = None

    def estimate_interval(self, model, features: np.ndarray,
                         confidence_level: float = 0.95) -> Tuple[float, float, float]:
        """
        Estimate prediction interval using bootstrap

        Args:
            model: Trained model with predict method
            features: Input features
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            (lower_bound, prediction, upper_bound)
        """
        # This would require access to multiple model checkpoints or dropout-based uncertainty
        # For now, return simple bounds based on calibration
        # In production, implement MC Dropout or ensemble-based uncertainty

        logger.warning("Confidence intervals not fully implemented - using simple bounds")

        prediction = float(model(features).item())

        # Simple heuristic: 10% around prediction
        margin = 0.1
        lower = max(0.0, prediction - margin)
        upper = min(1.0, prediction + margin)

        return lower, prediction, upper


class KellyPositionSizer:
    """
    Calculate optimal position size using Kelly Criterion

    Kelly Criterion maximizes long-term growth rate while accounting for edge and risk
    """

    def __init__(self, kelly_fraction: float = 0.25):
        """
        Args:
            kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
                           Fractional Kelly reduces volatility and risk of ruin
        """
        self.kelly_fraction = kelly_fraction

    def calculate_position_size(self, win_prob: float, avg_win: float,
                                avg_loss: float) -> float:
        """
        Calculate Kelly position size

        Kelly formula: f = (bp - q) / b
        where:
            f = fraction of capital to bet
            b = ratio of win/loss (avg_win / avg_loss)
            p = probability of winning
            q = probability of losing (1 - p)

        Args:
            win_prob: Calibrated probability of winning trade
            avg_win: Average winning trade size (in ticks or $)
            avg_loss: Average losing trade size (in ticks or $)

        Returns:
            Position size as fraction of max position (0-1)
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0

        if avg_loss <= 0 or avg_win <= 0:
            return 0.0

        # Kelly calculation
        b = avg_win / avg_loss  # Win/loss ratio
        p = win_prob
        q = 1 - p

        kelly = (b * p - q) / b

        # Only bet if positive expectation
        if kelly <= 0:
            return 0.0

        # Use fractional Kelly for safety
        fractional_kelly = kelly * self.kelly_fraction

        # Cap at 100%
        position_size = min(fractional_kelly, 1.0)

        return float(position_size)

    def calculate_expected_value(self, win_prob: float, avg_win: float,
                                avg_loss: float) -> float:
        """
        Calculate expected value of trade

        EV = (win_prob * avg_win) - ((1 - win_prob) * avg_loss)

        Args:
            win_prob: Probability of winning
            avg_win: Average win
            avg_loss: Average loss

        Returns:
            Expected value (positive = profitable)
        """
        ev = (win_prob * avg_win) - ((1 - win_prob) * avg_loss)
        return float(ev)

    def get_optimal_threshold(self, calibrated_probs: np.ndarray,
                             outcomes: np.ndarray, avg_win: float,
                             avg_loss: float) -> float:
        """
        Find optimal probability threshold for entry

        Scans different thresholds to find which maximizes Kelly position size

        Args:
            calibrated_probs: Calibrated probability predictions
            outcomes: Actual outcomes (0 or 1)
            avg_win: Average winning trade
            avg_loss: Average losing trade

        Returns:
            Optimal threshold probability
        """
        thresholds = np.arange(0.5, 0.8, 0.05)
        best_threshold = 0.6
        best_kelly = 0.0

        for threshold in thresholds:
            # Filter trades above threshold
            mask = calibrated_probs >= threshold

            if np.sum(mask) < 10:  # Need minimum samples
                continue

            # Calculate win rate at this threshold
            win_rate = np.mean(outcomes[mask])

            # Calculate Kelly
            kelly = self.calculate_position_size(win_rate, avg_win, avg_loss)

            if kelly > best_kelly:
                best_kelly = kelly
                best_threshold = threshold

        logger.info(f"Optimal threshold: {best_threshold:.3f} with Kelly size: {best_kelly:.3f}")

        return float(best_threshold)
