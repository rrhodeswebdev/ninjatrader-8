"""
Curriculum Learning for Trading Model

Trains model progressively from easy to hard examples:
1. Start with clear trending/ranging periods
2. Gradually add more complex market conditions
3. Finally include choppy/difficult markets

This improves convergence speed and final performance.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from model import detect_market_regime, calculate_adx


class CurriculumScheduler:
    """
    Schedule training from easy to hard examples

    Difficulty criteria:
    - Volatility (lower = easier)
    - Trend clarity (stronger trend = easier)
    - Market regime (trending = easier than ranging)
    """

    def __init__(self, n_stages: int = 4):
        """
        Initialize curriculum scheduler

        Args:
            n_stages: Number of difficulty stages (default: 4)
        """
        self.n_stages = n_stages
        self.difficulty_scores = None
        self.stage_thresholds = None

    def calculate_difficulty_scores(self, df: pd.DataFrame,
                                    lookback: int = 50) -> np.ndarray:
        """
        Calculate difficulty score for each sample

        Lower score = easier (clear patterns)
        Higher score = harder (noisy, choppy)

        Args:
            df: DataFrame with OHLCV data
            lookback: Window for difficulty calculation

        Returns:
            Array of difficulty scores (one per bar)
        """
        print("\n" + "="*70)
        print("CALCULATING SAMPLE DIFFICULTY")
        print("="*70)

        n_bars = len(df)
        difficulty_scores = np.zeros(n_bars)

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        for i in range(lookback, n_bars):
            window_data = df.iloc[i-lookback:i+1]

            # Factor 1: Volatility (normalized)
            returns = np.diff(close[i-lookback:i+1]) / close[i-lookback:i]
            volatility = np.std(returns)
            volatility_score = min(volatility / 0.02, 2.0)  # Normalize, cap at 2.0

            # Factor 2: Trend clarity (via ADX)
            try:
                from model import calculate_adx
                adx = calculate_adx(
                    high[i-lookback:i+1],
                    low[i-lookback:i+1],
                    close[i-lookback:i+1],
                    period=14
                )
                current_adx = adx[-1] if len(adx) > 0 else 20

                # Lower ADX = harder (ranging market)
                trend_clarity_score = (30 - current_adx) / 30  # Normalize, 0 = clear trend, 1 = no trend
                trend_clarity_score = max(0, min(trend_clarity_score, 1.0))
            except:
                trend_clarity_score = 0.5

            # Factor 3: Regime difficulty
            regime = detect_market_regime(window_data, lookback=min(lookback, len(window_data)-1))
            regime_difficulty = {
                'trending_normal': 0.2,
                'trending_high_vol': 0.4,
                'ranging_low_vol': 0.6,
                'ranging_normal': 0.7,
                'high_vol_chaos': 1.0,
                'transitional': 0.8,
                'unknown': 0.5
            }
            regime_score = regime_difficulty.get(regime, 0.5)

            # Factor 4: Price action complexity (number of direction changes)
            direction_changes = 0
            for j in range(i-lookback+1, i):
                if (close[j] - close[j-1]) * (close[j+1] - close[j]) < 0:
                    direction_changes += 1
            complexity_score = min(direction_changes / 20, 1.0)  # Normalize

            # Combine factors (weighted average)
            difficulty = (
                0.25 * volatility_score +
                0.30 * trend_clarity_score +
                0.30 * regime_score +
                0.15 * complexity_score
            )

            difficulty_scores[i] = difficulty

        # Fill initial bars with mean difficulty
        if lookback > 0:
            difficulty_scores[:lookback] = np.mean(difficulty_scores[lookback:])

        self.difficulty_scores = difficulty_scores

        # Calculate stage thresholds (percentiles)
        self.stage_thresholds = [
            np.percentile(difficulty_scores, p)
            for p in np.linspace(0, 100, self.n_stages + 1)
        ]

        print(f"\nDifficulty Score Statistics:")
        print(f"  Min:  {np.min(difficulty_scores):.3f}")
        print(f"  Mean: {np.mean(difficulty_scores):.3f}")
        print(f"  Max:  {np.max(difficulty_scores):.3f}")
        print(f"\nStage Thresholds:")
        for i in range(self.n_stages):
            print(f"  Stage {i+1}: {self.stage_thresholds[i]:.3f} - {self.stage_thresholds[i+1]:.3f}")

        return difficulty_scores

    def get_samples_for_stage(self, stage: int, all_indices: np.ndarray) -> np.ndarray:
        """
        Get sample indices for a given curriculum stage

        Args:
            stage: Stage number (0-indexed)
            all_indices: All available indices

        Returns:
            Indices for this stage
        """
        if self.difficulty_scores is None:
            raise ValueError("Calculate difficulty scores first with calculate_difficulty_scores()")

        if stage >= self.n_stages:
            # Final stage: use all samples
            return all_indices

        # Get threshold range for this stage
        min_difficulty = self.stage_thresholds[0]
        max_difficulty = self.stage_thresholds[stage + 1]

        # Get indices within difficulty range
        stage_mask = (self.difficulty_scores[all_indices] >= min_difficulty) & \
                    (self.difficulty_scores[all_indices] <= max_difficulty)

        stage_indices = all_indices[stage_mask]

        return stage_indices

    def get_curriculum_schedule(self, total_epochs: int) -> List[Tuple[int, int, float]]:
        """
        Generate curriculum schedule: (start_epoch, end_epoch, max_difficulty)

        Args:
            total_epochs: Total training epochs

        Returns:
            List of (start_epoch, end_epoch, max_difficulty_threshold)
        """
        epochs_per_stage = total_epochs // self.n_stages

        schedule = []
        for stage in range(self.n_stages):
            start_epoch = stage * epochs_per_stage
            end_epoch = (stage + 1) * epochs_per_stage if stage < self.n_stages - 1 else total_epochs
            max_difficulty = self.stage_thresholds[stage + 1]

            schedule.append((start_epoch, end_epoch, max_difficulty))

        return schedule


def train_with_curriculum(model, df: pd.DataFrame,
                         total_epochs: int = 100,
                         n_stages: int = 4,
                         **train_kwargs) -> Dict:
    """
    Train model using curriculum learning

    Args:
        model: TradingModel instance
        df: Training DataFrame
        total_epochs: Total epochs to train
        n_stages: Number of curriculum stages
        **train_kwargs: Additional arguments for model.train()

    Returns:
        Training results dictionary
    """
    print("\n" + "="*70)
    print("CURRICULUM LEARNING")
    print("="*70)

    # Initialize curriculum
    scheduler = CurriculumScheduler(n_stages=n_stages)

    # Calculate difficulty scores
    difficulty_scores = scheduler.calculate_difficulty_scores(df, lookback=50)

    # Get curriculum schedule
    schedule = scheduler.get_curriculum_schedule(total_epochs)

    print(f"\nCurriculum Schedule ({n_stages} stages, {total_epochs} total epochs):")
    for i, (start, end, max_diff) in enumerate(schedule):
        n_epochs = end - start
        print(f"  Stage {i+1}: Epochs {start:3d}-{end:3d} ({n_epochs:2d} epochs) - Max difficulty: {max_diff:.3f}")

    # Training history
    history = {
        'stage_results': [],
        'all_losses': [],
        'all_accuracies': [],
        'all_sharpe_ratios': []
    }

    # Train each stage
    for stage_idx, (start_epoch, end_epoch, max_difficulty) in enumerate(schedule):
        print(f"\n{'='*70}")
        print(f"STAGE {stage_idx + 1}/{n_stages}")
        print(f"Epochs {start_epoch}-{end_epoch}, Max difficulty: {max_difficulty:.3f}")
        print('='*70)

        # Filter dataset by difficulty
        stage_mask = difficulty_scores <= max_difficulty
        n_samples_stage = np.sum(stage_mask)

        print(f"Using {n_samples_stage}/{len(df)} samples ({n_samples_stage/len(df)*100:.1f}%)")

        # Create stage dataset
        df_stage = df[stage_mask].reset_index(drop=True)

        # Train on this stage
        stage_epochs = end_epoch - start_epoch

        # Update model's starting epoch for continuity
        if hasattr(model, 'current_epoch'):
            model.current_epoch = start_epoch

        try:
            stage_results = model.train(
                df_stage,
                epochs=stage_epochs,
                **train_kwargs
            )

            history['stage_results'].append({
                'stage': stage_idx + 1,
                'epochs': stage_epochs,
                'n_samples': n_samples_stage,
                'max_difficulty': max_difficulty,
                'results': stage_results
            })

            print(f"\n✓ Stage {stage_idx + 1} complete")

        except Exception as e:
            print(f"\n❌ Stage {stage_idx + 1} failed: {e}")
            continue

    print(f"\n{'='*70}")
    print("CURRICULUM TRAINING COMPLETE")
    print('='*70)

    # Final evaluation on full dataset
    print("\nFinal evaluation on full dataset...")
    try:
        signal, confidence = model.predict(df)
        print(f"  Final model can make predictions ✓")
    except Exception as e:
        print(f"  ⚠️  Final model prediction failed: {e}")

    return history


def adaptive_curriculum_learning(model, df: pd.DataFrame,
                                 total_epochs: int = 100,
                                 performance_threshold: float = 0.7) -> Dict:
    """
    Adaptive curriculum learning - progress to next stage only when ready

    Args:
        model: TradingModel instance
        df: Training DataFrame
        total_epochs: Maximum epochs to train
        performance_threshold: Accuracy threshold to advance (0-1)

    Returns:
        Training results
    """
    print("\n" + "="*70)
    print("ADAPTIVE CURRICULUM LEARNING")
    print("="*70)

    scheduler = CurriculumScheduler(n_stages=4)
    difficulty_scores = scheduler.calculate_difficulty_scores(df, lookback=50)

    current_stage = 0
    current_epoch = 0
    history = []

    while current_epoch < total_epochs and current_stage < scheduler.n_stages:
        max_difficulty = scheduler.stage_thresholds[current_stage + 1]

        print(f"\n{'='*70}")
        print(f"STAGE {current_stage + 1} - Max difficulty: {max_difficulty:.3f}")
        print('='*70)

        # Filter samples
        stage_mask = difficulty_scores <= max_difficulty
        df_stage = df[stage_mask].reset_index(drop=True)

        # Train for 10 epochs
        stage_epochs = min(10, total_epochs - current_epoch)

        results = model.train(df_stage, epochs=stage_epochs, verbose=True)

        current_epoch += stage_epochs

        # Check if model is ready to advance
        val_accuracy = results.get('validation_accuracy', 0.0) if isinstance(results, dict) else 0.0

        print(f"\nStage {current_stage + 1} Validation Accuracy: {val_accuracy:.3f}")

        if val_accuracy >= performance_threshold:
            print(f"✓ Advancing to next stage (threshold {performance_threshold:.3f} met)")
            current_stage += 1
        else:
            print(f"⚠️  Accuracy below threshold, continuing stage {current_stage + 1}")

        history.append({
            'stage': current_stage,
            'epoch': current_epoch,
            'val_accuracy': val_accuracy,
            'advanced': val_accuracy >= performance_threshold
        })

    print("\n" + "="*70)
    print("ADAPTIVE CURRICULUM COMPLETE")
    print("="*70)

    return {'history': history, 'final_stage': current_stage}


if __name__ == '__main__':
    print("Curriculum Learning Module")
    print("="*70)
    print("\nThis module implements curriculum learning for trading models")
    print("\nUsage:")
    print("  from curriculum_learning import train_with_curriculum")
    print("  from model import TradingModel")
    print()
    print("  model = TradingModel()")
    print("  results = train_with_curriculum(model, df, total_epochs=100, n_stages=4)")
