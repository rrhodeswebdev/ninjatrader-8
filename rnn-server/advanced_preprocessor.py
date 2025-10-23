"""
Advanced Data Preprocessing Pipeline - Complete Implementation

Professional-grade data preprocessing including:
- Missing value handling
- Outlier detection and clipping
- Robust feature scaling
- Temporal alignment
- Data quality validation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Optional, Dict
import warnings


class AdvancedDataPreprocessor:
    """
    Professional-grade data preprocessing pipeline
    """

    def __init__(self):
        self.scaler = None
        self.outlier_bounds = {}
        self.feature_stats = {}

    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        fit: bool = False
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline

        Args:
            df: DataFrame to preprocess
            fit: Whether to fit transformers (True for training data)

        Returns:
            Preprocessed DataFrame
        """
        # 1. Handle missing values
        df = self._handle_missing_values(df)

        # 2. Detect and handle outliers
        df = self._handle_outliers(df, fit=fit)

        # 3. Feature scaling (robust to outliers)
        df = self._scale_features(df, fit=fit)

        # 4. Temporal alignment
        df = self._align_temporal_features(df)

        # 5. Data quality validation
        if fit:
            self._validate_data_quality(df)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligent missing value handling

        Args:
            df: DataFrame with potential missing values

        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()

        for col in df.columns:
            if df[col].isna().any():
                # Forward fill for prices (carry last known price)
                if any(price_col in col.lower() for price_col in ['price', 'open', 'high', 'low', 'close', 'vwap', 'poc']):
                    df[col] = df[col].fillna(method='ffill')

                # Zero fill for volume (no volume = no trades)
                elif 'volume' in col.lower():
                    df[col] = df[col].fillna(0)

                # Zero fill for boolean/binary features
                elif df[col].dtype == bool or set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
                    df[col] = df[col].fillna(0)

                # Interpolate for continuous features
                else:
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')

        # Drop rows still containing NaN (beginning of series)
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)

        if dropped > 0:
            warnings.warn(f"Dropped {dropped} rows with remaining NaN values")

        return df

    def _handle_outliers(
        self,
        df: pd.DataFrame,
        fit: bool = False,
        method: str = 'iqr'
    ) -> pd.DataFrame:
        """
        Detect and clip outliers using IQR or Z-score method

        Args:
            df: DataFrame
            fit: Whether to calculate new bounds
            method: 'iqr' or 'zscore'

        Returns:
            DataFrame with outliers clipped
        """
        df = df.copy()

        if fit:
            self.outlier_bounds = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
            else:  # z-score
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - 4 * std
                upper_bound = mean + 4 * std

            if fit:
                self.outlier_bounds[col] = (lower_bound, upper_bound)

            # Clip outliers
            if col in self.outlier_bounds:
                df[col] = df[col].clip(
                    self.outlier_bounds[col][0],
                    self.outlier_bounds[col][1]
                )

        return df

    def _scale_features(
        self,
        df: pd.DataFrame,
        fit: bool = False
    ) -> pd.DataFrame:
        """
        Robust feature scaling using RobustScaler

        RobustScaler is resistant to outliers (uses median and IQR)

        Args:
            df: DataFrame
            fit: Whether to fit the scaler

        Returns:
            Scaled DataFrame
        """
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if not numeric_cols:
            return df

        if fit:
            self.scaler = RobustScaler()
            scaled_values = self.scaler.fit_transform(df[numeric_cols])

            # Store feature statistics
            for i, col in enumerate(numeric_cols):
                self.feature_stats[col] = {
                    'median': df[col].median(),
                    'iqr': df[col].quantile(0.75) - df[col].quantile(0.25)
                }
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            scaled_values = self.scaler.transform(df[numeric_cols])

        # Create new dataframe with scaled values
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scaled_values

        return df_scaled

    def _align_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure temporal features are properly aligned

        Args:
            df: DataFrame

        Returns:
            Temporally aligned DataFrame
        """
        df = df.copy()

        # Sort by time if time column exists
        if 'time' in df.columns:
            df = df.sort_values('time').reset_index(drop=True)

        return df

    def _validate_data_quality(self, df: pd.DataFrame):
        """
        Validate data quality and warn of issues

        Args:
            df: DataFrame to validate
        """
        print("\n" + "="*70)
        print("DATA QUALITY VALIDATION")
        print("="*70)

        # Check for remaining NaNs
        nan_counts = df.isna().sum()
        if nan_counts.any():
            print("\n⚠️  Remaining NaN values:")
            print(nan_counts[nan_counts > 0])

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count

        if inf_counts:
            print("\n⚠️  Infinite values found:")
            for col, count in inf_counts.items():
                print(f"  {col}: {count}")

        # Check for constant columns (no variance)
        constant_cols = []
        for col in numeric_cols:
            if df[col].std() == 0:
                constant_cols.append(col)

        if constant_cols:
            print("\n⚠️  Constant columns (no variance):")
            for col in constant_cols:
                print(f"  {col}")

        # Check data distribution
        print("\n✅ Data shape:", df.shape)
        print("✅ Numeric features:", len(numeric_cols))
        print("✅ Non-numeric features:", len(df.columns) - len(numeric_cols))

        print("="*70 + "\n")

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        feature_name: str
    ) -> np.ndarray:
        """
        Inverse transform scaled predictions back to original scale

        Args:
            predictions: Scaled predictions
            feature_name: Name of the feature to inverse transform

        Returns:
            Predictions in original scale
        """
        if self.scaler is None or feature_name not in self.feature_stats:
            return predictions

        # Get feature statistics
        median = self.feature_stats[feature_name]['median']
        iqr = self.feature_stats[feature_name]['iqr']

        # Inverse RobustScaler transformation
        # RobustScaler: (X - median) / IQR
        # Inverse: X * IQR + median
        original_scale = predictions * iqr + median

        return original_scale

    def get_feature_importance_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate feature importance weights based on variance

        Higher variance features get higher weights

        Args:
            df: DataFrame with features

        Returns:
            Dict of feature: weight
        """
        weights = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            variance = df[col].var()
            weights[col] = float(variance)

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}

        return weights


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str = 'signal',
    sequence_length: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for training

    Args:
        df: Preprocessed DataFrame with features
        target_column: Name of target column
        sequence_length: Length of input sequences

    Returns:
        Tuple of (X, y) arrays
    """
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_column and col != 'time']
    X_data = df[feature_cols].values
    y_data = df[target_column].values

    # Create sequences
    X_sequences = []
    y_sequences = []

    for i in range(len(X_data) - sequence_length):
        X_sequences.append(X_data[i:i+sequence_length])
        y_sequences.append(y_data[i+sequence_length])

    X = np.array(X_sequences)
    y = np.array(y_sequences)

    return X, y
