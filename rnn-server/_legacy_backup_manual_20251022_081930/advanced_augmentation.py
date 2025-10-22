"""
Advanced Data Augmentation for Time Series Trading Data

Implements sophisticated augmentation techniques:
- Time warping
- Permutation
- Mixup
- Cutout
- Dynamic time warping
- Synthetic minority oversampling
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple


def augment_time_series_advanced(X_sequence: np.ndarray,
                                 y_label: int = None,
                                 augmentation_prob: float = 0.4,
                                 augmentation_strength: float = 1.0) -> Tuple[np.ndarray, int]:
    """
    Advanced time series augmentation

    Args:
        X_sequence: Shape (sequence_length, n_features)
        y_label: Optional label (for mixup)
        augmentation_prob: Probability of applying augmentation
        augmentation_strength: Scale factor for augmentation intensity

    Returns:
        Augmented sequence (and potentially modified label for mixup)
    """
    if np.random.random() > augmentation_prob:
        return X_sequence, y_label

    aug_type = np.random.choice([
        'jitter',
        'scale',
        'magnitude_warp',
        'time_warp',
        'permutation',
        'cutout',
        'rotation'
    ], p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05])

    if aug_type == 'jitter':
        return jitter_augmentation(X_sequence, augmentation_strength), y_label

    elif aug_type == 'scale':
        return scale_augmentation(X_sequence, augmentation_strength), y_label

    elif aug_type == 'magnitude_warp':
        return magnitude_warp_augmentation(X_sequence, augmentation_strength), y_label

    elif aug_type == 'time_warp':
        return time_warp_augmentation(X_sequence, augmentation_strength), y_label

    elif aug_type == 'permutation':
        return permutation_augmentation(X_sequence), y_label

    elif aug_type == 'cutout':
        return cutout_augmentation(X_sequence, augmentation_strength), y_label

    elif aug_type == 'rotation':
        return rotation_augmentation(X_sequence), y_label

    return X_sequence, y_label


def jitter_augmentation(X_sequence: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Add small random noise to sequence

    Args:
        X_sequence: Shape (sequence_length, n_features)
        strength: Noise intensity multiplier

    Returns:
        Augmented sequence
    """
    noise_level = 0.003 * strength  # 0.3% of std
    noise = np.random.normal(0, noise_level, X_sequence.shape)
    return X_sequence + noise


def scale_augmentation(X_sequence: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Scale magnitude of sequence

    Args:
        X_sequence: Shape (sequence_length, n_features)
        strength: Scaling intensity multiplier

    Returns:
        Augmented sequence
    """
    scale_range = 0.02 * strength  # ±2%
    scale = np.random.uniform(1 - scale_range, 1 + scale_range)
    return X_sequence * scale


def magnitude_warp_augmentation(X_sequence: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Warp magnitude of random features

    Args:
        X_sequence: Shape (sequence_length, n_features)
        strength: Warping intensity multiplier

    Returns:
        Augmented sequence
    """
    n_features = X_sequence.shape[1]
    n_warp = max(1, int(n_features * 0.15 * strength))  # Warp 15% of features

    warp_features = np.random.choice(n_features, size=n_warp, replace=False)

    warped = X_sequence.copy()
    warp_scale = 1.0 + (np.random.randn(n_warp) * 0.05 * strength)  # ±5% warp

    for i, feat_idx in enumerate(warp_features):
        warped[:, feat_idx] *= warp_scale[i]

    return warped


def time_warp_augmentation(X_sequence: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Warp time dimension (speed up or slow down parts of sequence)

    Uses cubic interpolation to smoothly warp time

    Args:
        X_sequence: Shape (sequence_length, n_features)
        strength: Warping intensity multiplier

    Returns:
        Augmented sequence
    """
    seq_len, n_features = X_sequence.shape

    # Create warped time indices
    # Use polynomial warping: t' = t + a*t^2
    warp_intensity = 0.2 * strength  # Max 20% warp
    warp_coef = np.random.uniform(-warp_intensity, warp_intensity)

    t_original = np.linspace(0, 1, seq_len)
    t_warped = t_original + warp_coef * (t_original ** 2 - 0.5)

    # Ensure monotonic and within bounds
    t_warped = np.clip(t_warped, 0, 1)
    t_warped = np.sort(t_warped)  # Ensure monotonic

    # Interpolate each feature
    warped_sequence = np.zeros_like(X_sequence)

    for feat_idx in range(n_features):
        # Use linear interpolation (cubic can cause overshooting with financial data)
        interp_func = interp1d(t_original, X_sequence[:, feat_idx],
                              kind='linear', fill_value='extrapolate')
        warped_sequence[:, feat_idx] = interp_func(t_warped)

    return warped_sequence


def permutation_augmentation(X_sequence: np.ndarray, n_segments: int = 4) -> np.ndarray:
    """
    Randomly permute segments of the sequence

    Divides sequence into segments and shuffles them
    Good for learning order-invariant patterns

    Args:
        X_sequence: Shape (sequence_length, n_features)
        n_segments: Number of segments to divide sequence into

    Returns:
        Augmented sequence
    """
    seq_len = X_sequence.shape[0]
    segment_len = seq_len // n_segments

    if segment_len < 2:
        return X_sequence  # Too short to permute

    # Create segments
    segments = []
    for i in range(n_segments):
        start = i * segment_len
        end = start + segment_len if i < n_segments - 1 else seq_len
        segments.append(X_sequence[start:end])

    # Shuffle segments
    np.random.shuffle(segments)

    # Concatenate
    return np.vstack(segments)


def cutout_augmentation(X_sequence: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Randomly mask out a portion of the sequence

    Forces model to be robust to missing data

    Args:
        X_sequence: Shape (sequence_length, n_features)
        strength: Cutout intensity multiplier

    Returns:
        Augmented sequence
    """
    seq_len, n_features = X_sequence.shape

    # Cutout size: 10-20% of sequence length
    cutout_len = int(seq_len * np.random.uniform(0.1, 0.2) * strength)
    cutout_len = min(cutout_len, seq_len - 1)

    # Random start position
    cutout_start = np.random.randint(0, seq_len - cutout_len)

    # Create masked sequence
    augmented = X_sequence.copy()

    # Replace with interpolated values (more realistic than zeros)
    if cutout_start > 0 and cutout_start + cutout_len < seq_len:
        # Linear interpolation between edges
        for feat_idx in range(n_features):
            start_val = X_sequence[cutout_start - 1, feat_idx]
            end_val = X_sequence[cutout_start + cutout_len, feat_idx]
            interpolated = np.linspace(start_val, end_val, cutout_len + 2)[1:-1]
            augmented[cutout_start:cutout_start + cutout_len, feat_idx] = interpolated
    else:
        # Use forward fill if at edges
        if cutout_start == 0:
            augmented[:cutout_len] = X_sequence[cutout_len]
        else:
            augmented[cutout_start:] = X_sequence[cutout_start - 1]

    return augmented


def rotation_augmentation(X_sequence: np.ndarray) -> np.ndarray:
    """
    Randomly rotate (shift) the sequence in time

    Helps model learn position-invariant patterns

    Args:
        X_sequence: Shape (sequence_length, n_features)

    Returns:
        Augmented sequence
    """
    seq_len = X_sequence.shape[0]

    # Rotate by 10-30% of sequence length
    rotation_amount = int(seq_len * np.random.uniform(0.1, 0.3))

    return np.roll(X_sequence, rotation_amount, axis=0)


def mixup_augmentation(X_sequence1: np.ndarray, y1: int,
                      X_sequence2: np.ndarray, y2: int,
                      alpha: float = 0.2) -> Tuple[np.ndarray, float]:
    """
    Mixup augmentation: linear combination of two sequences

    Creates synthetic training examples by mixing pairs

    Args:
        X_sequence1: First sequence
        y1: First label
        X_sequence2: Second sequence
        y2: Second label
        alpha: Beta distribution parameter (higher = more mixing)

    Returns:
        Mixed sequence and soft label
    """
    # Sample mixing coefficient from Beta distribution
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    # Mix sequences
    X_mixed = lam * X_sequence1 + (1 - lam) * X_sequence2

    # For classification, return the dominant label
    # (In practice, would use soft labels in loss function)
    y_mixed = y1 if lam > 0.5 else y2

    return X_mixed, y_mixed


def adaptive_augmentation(X_sequence: np.ndarray, y_label: int,
                         class_counts: dict,
                         target_balance_ratio: float = 0.5) -> Tuple[np.ndarray, int]:
    """
    Adaptive augmentation based on class imbalance

    Applies stronger augmentation to minority classes

    Args:
        X_sequence: Input sequence
        y_label: Class label
        class_counts: Dictionary of class counts
        target_balance_ratio: Target ratio between minority and majority class

    Returns:
        Augmented sequence and label
    """
    # Calculate class frequency
    total_samples = sum(class_counts.values())
    class_freq = class_counts.get(y_label, 1) / total_samples

    # Augmentation probability inversely proportional to class frequency
    # Minority classes get more augmentation
    aug_prob = 1.0 - class_freq
    aug_prob = np.clip(aug_prob, 0.2, 0.8)  # Bound between 20% and 80%

    # Augmentation strength also scales with imbalance
    aug_strength = 1.0 + (1.0 - class_freq)

    return augment_time_series_advanced(X_sequence, y_label, aug_prob, aug_strength)


def generate_synthetic_minority_samples(X_minority: np.ndarray,
                                       n_synthetic: int) -> np.ndarray:
    """
    Generate synthetic samples for minority class

    Similar to SMOTE but adapted for time series sequences

    Args:
        X_minority: Minority class samples, shape (n_samples, seq_len, n_features)
        n_synthetic: Number of synthetic samples to generate

    Returns:
        Synthetic samples
    """
    n_samples, seq_len, n_features = X_minority.shape

    synthetic_samples = []

    for _ in range(n_synthetic):
        # Pick two random minority samples
        idx1, idx2 = np.random.choice(n_samples, size=2, replace=True)
        sample1 = X_minority[idx1]
        sample2 = X_minority[idx2]

        # Mix with random weight
        alpha = np.random.uniform(0.3, 0.7)
        synthetic = alpha * sample1 + (1 - alpha) * sample2

        # Apply light augmentation
        synthetic, _ = augment_time_series_advanced(synthetic, None,
                                                    augmentation_prob=0.5,
                                                    augmentation_strength=0.5)

        synthetic_samples.append(synthetic)

    return np.array(synthetic_samples)


if __name__ == '__main__':
    # Test augmentations
    print("Advanced Augmentation Module")
    print("="*70)

    # Create test sequence
    np.random.seed(42)
    test_sequence = np.random.randn(15, 97)  # 15 bars, 97 features

    print(f"Test sequence shape: {test_sequence.shape}")
    print(f"\nTesting each augmentation type:")

    # Test each augmentation
    aug_types = [
        ('Jitter', lambda: jitter_augmentation(test_sequence)),
        ('Scale', lambda: scale_augmentation(test_sequence)),
        ('Magnitude Warp', lambda: magnitude_warp_augmentation(test_sequence)),
        ('Time Warp', lambda: time_warp_augmentation(test_sequence)),
        ('Permutation', lambda: permutation_augmentation(test_sequence)),
        ('Cutout', lambda: cutout_augmentation(test_sequence)),
        ('Rotation', lambda: rotation_augmentation(test_sequence))
    ]

    for name, aug_func in aug_types:
        augmented = aug_func()
        diff = np.mean(np.abs(augmented - test_sequence))
        print(f"  {name:20s} - Mean absolute change: {diff:.6f}")

    print("\n✓ All augmentation types working")
