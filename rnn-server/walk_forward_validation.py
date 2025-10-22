"""
Walk-Forward Validation Module

Proper time series validation that prevents look-ahead bias.
Uses expanding or rolling windows to simulate realistic trading conditions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation for time series models

    Unlike random train/test splits, walk-forward validation:
    - Respects temporal ordering
    - Prevents look-ahead bias
    - Simulates realistic deployment conditions
    """

    def __init__(self,
                 train_days: int = 60,
                 test_days: int = 10,
                 step_days: int = 5,
                 min_train_samples: int = 1000):
        """
        Args:
            train_days: Days of training data
            test_days: Days of testing data
            step_days: Days to step forward for next split
            min_train_samples: Minimum samples required for training
        """
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.min_train_samples = min_train_samples

    def split_timeseries(self,
                        data: pd.DataFrame,
                        date_column: str = 'date') -> List[Dict]:
        """
        Generate walk-forward train/test splits

        Example:
        - Train: Days 1-60, Test: Days 61-70
        - Train: Days 6-65, Test: Days 66-75
        - Train: Days 11-70, Test: Days 71-80

        Args:
            data: DataFrame with time series data
            date_column: Name of date/datetime column

        Returns:
            List of split dictionaries with train/test indices
        """
        # Ensure data is sorted by date
        data = data.sort_values(date_column).reset_index(drop=True)

        dates = pd.to_datetime(data[date_column])
        unique_dates = sorted(dates.dt.date.unique())

        splits = []
        start_idx = 0

        while start_idx + self.train_days + self.test_days <= len(unique_dates):
            # Define date ranges
            train_start_date = unique_dates[start_idx]
            train_end_date = unique_dates[start_idx + self.train_days - 1]
            test_start_date = unique_dates[start_idx + self.train_days]
            test_end_date = unique_dates[min(start_idx + self.train_days + self.test_days - 1,
                                            len(unique_dates) - 1)]

            # Get indices for these date ranges
            train_mask = (dates.dt.date >= train_start_date) & (dates.dt.date <= train_end_date)
            test_mask = (dates.dt.date >= test_start_date) & (dates.dt.date <= test_end_date)

            train_indices = data[train_mask].index.tolist()
            test_indices = data[test_mask].index.tolist()

            # Ensure minimum training samples
            if len(train_indices) < self.min_train_samples:
                logger.warning(f"Skipping split - insufficient training samples: {len(train_indices)}")
                start_idx += self.step_days
                continue

            split = {
                'train_indices': train_indices,
                'test_indices': test_indices,
                'train_start': train_start_date,
                'train_end': train_end_date,
                'test_start': test_start_date,
                'test_end': test_end_date,
                'fold': len(splits)
            }

            splits.append(split)

            # Move forward
            start_idx += self.step_days

        logger.info(f"Generated {len(splits)} walk-forward splits")

        return splits

    def validate_model(self,
                      model,
                      data: pd.DataFrame,
                      features_columns: List[str],
                      target_column: str,
                      date_column: str = 'date',
                      fit_func = None,
                      predict_func = None) -> pd.DataFrame:
        """
        Perform walk-forward validation on a model

        Args:
            model: Model instance with fit/predict methods
            data: DataFrame with features and targets
            features_columns: List of feature column names
            target_column: Target column name
            date_column: Date column name
            fit_func: Custom fit function (optional)
            predict_func: Custom predict function (optional)

        Returns:
            DataFrame with validation results per fold
        """
        splits = self.split_timeseries(data, date_column)

        results = []

        for split in splits:
            fold = split['fold']
            logger.info(f"Fold {fold + 1}/{len(splits)}: "
                       f"Train {split['train_start']} to {split['train_end']}, "
                       f"Test {split['test_start']} to {split['test_end']}")

            # Get train and test data
            train_data = data.loc[split['train_indices']]
            test_data = data.loc[split['test_indices']]

            X_train = train_data[features_columns].values
            y_train = train_data[target_column].values

            X_test = test_data[features_columns].values
            y_test = test_data[target_column].values

            # Fit model on training data only
            if fit_func is not None:
                fit_func(model, X_train, y_train)
            else:
                model.fit(X_train, y_train)

            # Predict on test data (unseen future data)
            if predict_func is not None:
                predictions = predict_func(model, X_test)
            else:
                predictions = model.predict(X_test)

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, predictions)

            result = {
                'fold': fold,
                'train_start': split['train_start'],
                'train_end': split['train_end'],
                'test_start': split['test_start'],
                'test_end': split['test_end'],
                'train_samples': len(split['train_indices']),
                'test_samples': len(split['test_indices']),
                **metrics
            }

            results.append(result)

        results_df = pd.DataFrame(results)

        # Print summary
        self._print_summary(results_df)

        return results_df

    def _calculate_metrics(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate validation metrics

        Args:
            y_true: True labels
            y_pred: Predictions

        Returns:
            Dictionary of metrics
        """
        # Binary classification metrics
        if len(np.unique(y_true)) == 2:
            accuracy = np.mean((y_pred > 0.5) == y_true)

            # Precision and recall
            tp = np.sum((y_pred > 0.5) & (y_true == 1))
            fp = np.sum((y_pred > 0.5) & (y_true == 0))
            fn = np.sum((y_pred <= 0.5) & (y_true == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        else:
            # Regression metrics
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))

            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            }

    def _print_summary(self, results_df: pd.DataFrame):
        """Print validation summary"""
        print("\n" + "="*80)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("="*80)

        print(f"\nTotal Folds: {len(results_df)}")

        # Get metric columns (exclude metadata)
        metric_cols = [col for col in results_df.columns
                      if col not in ['fold', 'train_start', 'train_end',
                                    'test_start', 'test_end', 'train_samples', 'test_samples']]

        print("\nAverage Metrics Across Folds:")
        for col in metric_cols:
            mean_val = results_df[col].mean()
            std_val = results_df[col].std()
            print(f"  {col:15s}: {mean_val:.4f} ± {std_val:.4f}")

        print("\n" + "="*80 + "\n")


class ExpandingWindowValidator:
    """
    Expanding window validation (anchored walk-forward)

    Training window starts from beginning and expands,
    while test window slides forward.
    """

    def __init__(self,
                 initial_train_days: int = 60,
                 test_days: int = 10,
                 step_days: int = 5):
        """
        Args:
            initial_train_days: Initial training period
            test_days: Testing period
            step_days: Step size
        """
        self.initial_train_days = initial_train_days
        self.test_days = test_days
        self.step_days = step_days

    def split_timeseries(self,
                        data: pd.DataFrame,
                        date_column: str = 'date') -> List[Dict]:
        """
        Generate expanding window splits

        Example:
        - Train: Days 1-60, Test: Days 61-70
        - Train: Days 1-65, Test: Days 66-75
        - Train: Days 1-70, Test: Days 71-80

        Args:
            data: DataFrame with time series data
            date_column: Name of date column

        Returns:
            List of split dictionaries
        """
        data = data.sort_values(date_column).reset_index(drop=True)

        dates = pd.to_datetime(data[date_column])
        unique_dates = sorted(dates.dt.date.unique())

        splits = []
        anchor_date = unique_dates[0]  # Anchor to beginning
        test_start_idx = self.initial_train_days

        while test_start_idx + self.test_days <= len(unique_dates):
            train_end_date = unique_dates[test_start_idx - 1]
            test_start_date = unique_dates[test_start_idx]
            test_end_date = unique_dates[min(test_start_idx + self.test_days - 1,
                                            len(unique_dates) - 1)]

            # Training: from anchor to current point
            train_mask = (dates.dt.date >= anchor_date) & (dates.dt.date <= train_end_date)
            test_mask = (dates.dt.date >= test_start_date) & (dates.dt.date <= test_end_date)

            split = {
                'train_indices': data[train_mask].index.tolist(),
                'test_indices': data[test_mask].index.tolist(),
                'train_start': anchor_date,
                'train_end': train_end_date,
                'test_start': test_start_date,
                'test_end': test_end_date,
                'fold': len(splits)
            }

            splits.append(split)

            test_start_idx += self.step_days

        logger.info(f"Generated {len(splits)} expanding window splits")

        return splits


class PurgedKFold:
    """
    Purged K-Fold for time series with overlapping samples

    Used when training samples have temporal dependencies or
    when features use look-ahead windows.

    Purges samples near test set to prevent information leakage.
    """

    def __init__(self,
                 n_splits: int = 5,
                 purge_gap: int = 10):
        """
        Args:
            n_splits: Number of folds
            purge_gap: Number of samples to purge before/after test set
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap

    def split(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged k-fold splits

        Args:
            data: DataFrame

        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(data)
        indices = np.arange(n_samples)

        test_size = n_samples // self.n_splits
        splits = []

        for i in range(self.n_splits):
            # Test set
            test_start = i * test_size
            test_end = (i + 1) * test_size if i < self.n_splits - 1 else n_samples

            test_indices = indices[test_start:test_end]

            # Purge samples around test set
            purge_start = max(0, test_start - self.purge_gap)
            purge_end = min(n_samples, test_end + self.purge_gap)

            # Training set (excluding test and purge regions)
            train_mask = (indices < purge_start) | (indices >= purge_end)
            train_indices = indices[train_mask]

            splits.append((train_indices, test_indices))

        logger.info(f"Generated {len(splits)} purged k-fold splits")

        return splits
