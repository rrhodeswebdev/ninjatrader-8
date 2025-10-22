"""
Data Augmentation Module for Financial Time Series

Augmentation techniques specific to financial data that preserve
realistic market dynamics while increasing training sample diversity.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FinancialDataAugmentation:
    """
    Data augmentation techniques for financial time series

    These methods increase training samples without introducing
    unrealistic patterns that could harm model performance.
    """

    @staticmethod
    def jittering(data: np.ndarray, sigma: float = 0.03) -> np.ndarray:
        """
        Add small random noise to features

        Simulates measurement noise and minor market variations.
        Helps model generalize to small perturbations.

        Args:
            data: Input data of shape (batch, sequence, features)
            sigma: Standard deviation of noise (default 3%)

        Returns:
            Augmented data with same shape
        """
        noise = np.random.normal(0, sigma, data.shape)
        augmented = data + noise

        return augmented

    @staticmethod
    def scaling(data: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """
        Randomly scale features to simulate different volatility regimes

        Multiplies data by random factor to simulate high/low volatility periods.

        Args:
            data: Input data of shape (batch, sequence, features)
            sigma: Standard deviation of scaling factor (default 10%)

        Returns:
            Scaled data
        """
        scale_factor = np.random.normal(1.0, sigma)

        # Ensure scale factor is reasonable
        scale_factor = np.clip(scale_factor, 0.7, 1.3)

        augmented = data * scale_factor

        return augmented

    @staticmethod
    def time_warping(data: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """
        Warp time dimension to simulate different speeds of price movement

        Compresses or expands time axis to simulate fast/slow markets.
        Uses cubic spline interpolation for smooth warping.

        Args:
            data: Input data of shape (batch, sequence, features)
            sigma: Warping strength (default 0.2)

        Returns:
            Time-warped data with same shape
        """
        batch_size, seq_len, n_features = data.shape
        orig_steps = np.arange(seq_len)

        # Generate random time warp
        warp = np.cumsum(np.random.normal(1.0, sigma, seq_len))
        warp = warp / warp[-1] * (seq_len - 1)  # Normalize to original length

        warped = np.zeros_like(data)

        for b in range(batch_size):
            for f in range(n_features):
                try:
                    # Cubic spline interpolation
                    cs = CubicSpline(orig_steps, data[b, :, f])
                    warped[b, :, f] = cs(warp)
                except Exception:
                    # If interpolation fails, use original
                    warped[b, :, f] = data[b, :, f]

        return warped

    @staticmethod
    def window_slicing(data: np.ndarray, slice_ratio: float = 0.9) -> np.ndarray:
        """
        Use random subsequences of sequences

        Helps model learn from different time horizons.

        Args:
            data: Input data of shape (batch, sequence, features)
            slice_ratio: Fraction of sequence to keep (default 0.9)

        Returns:
            Sliced subsequence
        """
        batch_size, seq_len, n_features = data.shape
        slice_len = int(seq_len * slice_ratio)

        # Random start position
        start_idx = np.random.randint(0, seq_len - slice_len + 1)

        sliced = data[:, start_idx:start_idx + slice_len, :]

        return sliced

    @staticmethod
    def magnitude_warping(data: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """
        Smooth magnitude warping using cubic spline

        Changes magnitude of price movements while preserving shape.

        Args:
            data: Input data of shape (batch, sequence, features)
            sigma: Warping strength

        Returns:
            Magnitude-warped data
        """
        batch_size, seq_len, n_features = data.shape

        # Generate smooth warping curve
        knots = np.linspace(0, seq_len - 1, num=4)
        warp_values = np.random.normal(1.0, sigma, len(knots))

        # Create smooth curve
        cs = CubicSpline(knots, warp_values)
        warp_curve = cs(np.arange(seq_len))

        warped = np.zeros_like(data)

        for b in range(batch_size):
            for f in range(n_features):
                warped[b, :, f] = data[b, :, f] * warp_curve

        return warped

    @staticmethod
    def permutation(data: np.ndarray, n_segments: int = 4) -> np.ndarray:
        """
        Randomly permute segments of the sequence

        Breaks temporal dependencies while preserving local patterns.
        Use sparingly as it can disrupt important temporal structure.

        Args:
            data: Input data of shape (batch, sequence, features)
            n_segments: Number of segments to split and permute

        Returns:
            Permuted data
        """
        batch_size, seq_len, n_features = data.shape
        segment_len = seq_len // n_segments

        permuted = np.zeros_like(data)

        for b in range(batch_size):
            # Create random permutation
            perm = np.random.permutation(n_segments)

            for i, p in enumerate(perm):
                start_orig = p * segment_len
                end_orig = (p + 1) * segment_len if p < n_segments - 1 else seq_len

                start_new = i * segment_len
                end_new = (i + 1) * segment_len if i < n_segments - 1 else seq_len

                copy_len = min(end_orig - start_orig, end_new - start_new)

                permuted[b, start_new:start_new + copy_len, :] = \
                    data[b, start_orig:start_orig + copy_len, :]

        return permuted

    @staticmethod
    def mixup(data1: np.ndarray, data2: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """
        Mixup augmentation - linear interpolation between two samples

        Creates synthetic samples by blending two examples.

        Args:
            data1: First batch of data
            data2: Second batch of data
            alpha: Mixup coefficient (lower = more mixing)

        Returns:
            Mixed data
        """
        lam = np.random.beta(alpha, alpha)
        mixed = lam * data1 + (1 - lam) * data2

        return mixed


class AugmentationPipeline:
    """
    Configurable augmentation pipeline for financial data
    """

    def __init__(self,
                 techniques: Optional[list] = None,
                 probabilities: Optional[list] = None):
        """
        Args:
            techniques: List of augmentation technique names
            probabilities: Probability of applying each technique
        """
        self.augmentor = FinancialDataAugmentation()

        if techniques is None:
            techniques = ['jitter', 'scale']

        if probabilities is None:
            probabilities = [0.5] * len(techniques)

        self.techniques = techniques
        self.probabilities = probabilities

        logger.info(f"Initialized augmentation pipeline with techniques: {techniques}")

    def augment(self, data: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to data

        Args:
            data: Input data

        Returns:
            Augmented data
        """
        augmented = data.copy()

        for technique, prob in zip(self.techniques, self.probabilities):
            if np.random.random() < prob:
                augmented = self._apply_technique(augmented, technique)

        return augmented

    def _apply_technique(self, data: np.ndarray, technique: str) -> np.ndarray:
        """Apply specific augmentation technique"""

        if technique == 'jitter':
            return self.augmentor.jittering(data, sigma=0.03)

        elif technique == 'scale':
            return self.augmentor.scaling(data, sigma=0.1)

        elif technique == 'warp':
            return self.augmentor.time_warping(data, sigma=0.2)

        elif technique == 'slice':
            return self.augmentor.window_slicing(data, slice_ratio=0.9)

        elif technique == 'magnitude':
            return self.augmentor.magnitude_warping(data, sigma=0.2)

        else:
            logger.warning(f"Unknown augmentation technique: {technique}")
            return data


class RollingNormalizer:
    """
    Normalize features using only past data (no look-ahead bias)

    Critical for proper time series preprocessing to prevent data leakage.
    """

    def __init__(self, window: int = 252 * 390):
        """
        Args:
            window: Lookback window for normalization (~1 year of minute bars)
        """
        self.window = window
        self.mean_history = {}
        self.std_history = {}

    def fit_transform(self, feature_name: str, values: np.ndarray,
                     index: np.ndarray) -> np.ndarray:
        """
        Normalize using expanding window (only past data)

        Args:
            feature_name: Name of feature for tracking
            values: Feature values
            index: Time index for tracking statistics

        Returns:
            Normalized values
        """
        normalized = np.zeros_like(values, dtype=float)

        for i in range(len(values)):
            # Use only data up to current point
            start = max(0, i - self.window)
            historical = values[start:i + 1]

            if len(historical) < 2:
                normalized[i] = 0.0
                continue

            mean = np.mean(historical)
            std = np.std(historical)

            if std == 0:
                normalized[i] = 0.0
            else:
                normalized[i] = (values[i] - mean) / std

            # Store for inference
            self.mean_history[index[i]] = mean
            self.std_history[index[i]] = std

        return normalized

    def transform(self, feature_name: str, value: float,
                 current_time) -> float:
        """
        Normalize new value using most recent statistics

        Args:
            feature_name: Feature name
            value: New value to normalize
            current_time: Current timestamp

        Returns:
            Normalized value
        """
        if current_time not in self.mean_history:
            logger.warning(f"No normalization stats for {current_time}")
            return 0.0

        mean = self.mean_history[current_time]
        std = self.std_history[current_time]

        if std == 0:
            return 0.0

        normalized = (value - mean) / std

        return float(normalized)

    def save_stats(self, filepath: str):
        """Save normalization statistics"""
        import pickle

        with open(filepath, 'wb') as f:
            pickle.dump({
                'mean_history': self.mean_history,
                'std_history': self.std_history
            }, f)

        logger.info(f"Saved normalization stats to {filepath}")

    def load_stats(self, filepath: str):
        """Load normalization statistics"""
        import pickle

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.mean_history = data['mean_history']
        self.std_history = data['std_history']

        logger.info(f"Loaded normalization stats from {filepath}")
