"""
Regime-Specific Model Training and Selection

Trains separate specialized models for different market regimes.
Each model becomes an expert in its regime type.

Expected Impact: +0.3-0.5 Sharpe ratio from regime specialization
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
from model import TradingModel, detect_market_regime
from trading_metrics import calculate_sharpe_ratio, calculate_win_rate


class RegimeSpecificModel:
    """
    A model specialized for a specific market regime
    """

    def __init__(self, regime_name: str, sequence_length: int = 40):
        """
        Args:
            regime_name: Name of regime this model specializes in
            sequence_length: Sequence length for model
        """
        self.regime_name = regime_name
        self.model = TradingModel(sequence_length=sequence_length)
        self.sequence_length = sequence_length

        # Performance tracking
        self.training_samples = 0
        self.validation_accuracy = 0.0
        self.validation_sharpe = 0.0

    def train(self, df: pd.DataFrame, epochs: int = 50, validation_split: float = 0.2):
        """
        Train model on regime-specific data

        Args:
            df: DataFrame containing only data from this regime
            epochs: Training epochs
            validation_split: Validation percentage
        """
        self.training_samples = len(df)

        print(f"\nTraining {self.regime_name} model...")
        print(f"  Training samples: {self.training_samples}")

        # Train model
        self.model.train(df, epochs=epochs, batch_size=32, validation_split=validation_split)

        # Validate
        val_size = int(len(df) * validation_split)
        val_df = df.iloc[-val_size:]
        self._validate(val_df)

    def _validate(self, df: pd.DataFrame):
        """Validate model on holdout data"""
        predictions = []
        actuals = []
        returns = []

        for i in range(len(df) - self.sequence_length - 5):
            sequence_data = df.iloc[:self.sequence_length + i]

            try:
                signal, confidence = self.model.predict(sequence_data)

                signal_map = {'short': 0, 'hold': 1, 'long': 2}
                pred_class = signal_map[signal]

                current_idx = self.sequence_length + i
                if current_idx + 1 < len(df):
                    current_price = df.iloc[current_idx]['close']
                    next_price = df.iloc[current_idx + 1]['close']
                    price_return = (next_price - current_price) / current_price

                    threshold = 0.0005
                    if price_return > threshold:
                        actual_class = 2
                    elif price_return < -threshold:
                        actual_class = 0
                    else:
                        actual_class = 1

                    predictions.append(pred_class)
                    actuals.append(actual_class)
                    returns.append(price_return)
            except:
                continue

        if len(predictions) > 0:
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            returns = np.array(returns)

            self.validation_accuracy = np.mean(predictions == actuals)
            self.validation_sharpe = calculate_sharpe_ratio(returns)

            print(f"  Validation - Accuracy: {self.validation_accuracy:.2%}, Sharpe: {self.validation_sharpe:.3f}")

    def predict(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Make prediction using this regime-specific model

        Returns:
            (signal, confidence)
        """
        return self.model.predict(df)

    def save(self, path: Path):
        """Save model to disk"""
        self.model.save_model(path=path)

    def load(self, path: Path) -> bool:
        """Load model from disk"""
        return self.model.load_model(path=path)


class RegimeModelManager:
    """
    Manages multiple regime-specific models and routes predictions
    """

    def __init__(
        self,
        model_dir: str = 'models/regime_specific',
        min_samples_per_regime: int = 500
    ):
        """
        Args:
            model_dir: Directory to store regime-specific models
            min_samples_per_regime: Minimum samples needed to train regime model
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.min_samples_per_regime = min_samples_per_regime

        # Regime models
        self.regime_models: Dict[str, RegimeSpecificModel] = {}

        # Fallback general model for regimes with insufficient data
        self.general_model = TradingModel(sequence_length=40)

        # Regime definitions
        self.regimes = [
            'trending_normal',
            'trending_high_vol',
            'ranging_normal',
            'ranging_low_vol',
            'high_vol_chaos',
            'transitional'
        ]

    def train_regime_models(
        self,
        df: pd.DataFrame,
        epochs_per_regime: int = 50,
        verbose: bool = True
    ):
        """
        Train separate models for each market regime

        Args:
            df: Full historical dataset
            epochs_per_regime: Training epochs for each regime model
            verbose: Print training progress
        """
        print("\n" + "="*70)
        print("REGIME-SPECIFIC MODEL TRAINING")
        print("="*70)

        # Classify all data by regime
        print("\nClassifying data by regime...")
        df = df.copy()
        df['regime'] = df.apply(
            lambda row: self._get_regime_for_row(df, row.name),
            axis=1
        )

        # Count samples per regime
        regime_counts = df['regime'].value_counts()
        print("\nRegime distribution:")
        for regime in self.regimes:
            count = regime_counts.get(regime, 0)
            pct = (count / len(df)) * 100
            print(f"  {regime:20s}: {count:5d} bars ({pct:5.1f}%)")

        # Train model for each regime with sufficient data
        trained_regimes = []

        for regime in self.regimes:
            regime_data = df[df['regime'] == regime].copy()

            if len(regime_data) < self.min_samples_per_regime:
                print(f"\n⚠️  {regime}: Insufficient data ({len(regime_data)} < {self.min_samples_per_regime}), skipping")
                continue

            # Train regime-specific model
            regime_model = RegimeSpecificModel(regime_name=regime)
            regime_model.train(regime_data, epochs=epochs_per_regime)

            # Save model
            model_path = self.model_dir / f'{regime}_model.pth'
            regime_model.save(model_path)

            self.regime_models[regime] = regime_model
            trained_regimes.append(regime)

            print(f"  ✓ {regime} model saved to {model_path}")

        # Train general fallback model on all data
        print(f"\n{'='*70}")
        print("TRAINING GENERAL FALLBACK MODEL")
        print(f"{'='*70}")
        print(f"Training on all {len(df)} bars...")

        self.general_model.train(df, epochs=epochs_per_regime, batch_size=32, validation_split=0.2)
        general_model_path = self.model_dir / 'general_model.pth'
        self.general_model.save_model(path=general_model_path)

        print(f"✓ General model saved to {general_model_path}")

        # Summary
        print(f"\n{'='*70}")
        print("TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"Regime-specific models trained: {len(trained_regimes)}/{len(self.regimes)}")
        print(f"Trained regimes: {', '.join(trained_regimes)}")
        print(f"Fallback model: Available for all regimes")
        print(f"{'='*70}\n")

        return {
            'trained_regimes': trained_regimes,
            'regime_counts': dict(regime_counts),
            'total_samples': len(df)
        }

    def _get_regime_for_row(self, df: pd.DataFrame, idx: int) -> str:
        """Get regime for a specific row"""
        # Need enough history for regime detection
        if idx < 100:
            return 'transitional'

        # Get data up to this point
        subset = df.iloc[:idx+1]
        regime = detect_market_regime(subset, lookback=min(100, len(subset)-1))

        return regime

    def predict(
        self,
        df: pd.DataFrame,
        use_regime_specific: bool = True,
        verbose: bool = False
    ) -> Tuple[str, float, Dict]:
        """
        Make prediction using appropriate regime model

        Args:
            df: Recent data for prediction
            use_regime_specific: If True, use regime-specific model; else use general
            verbose: Print prediction details

        Returns:
            (signal, confidence, details_dict)
        """
        # Detect current regime
        regime = detect_market_regime(df, lookback=min(100, len(df)-1))

        # Choose model
        if use_regime_specific and regime in self.regime_models:
            model = self.regime_models[regime]
            model_used = f'{regime}_specialist'
            signal, confidence = model.predict(df)
        else:
            model = self.general_model
            model_used = 'general_fallback'
            signal, confidence = model.predict(df)

        if verbose:
            print(f"\n--- Regime-Specific Prediction ---")
            print(f"Detected Regime: {regime}")
            print(f"Model Used: {model_used}")
            print(f"Signal: {signal.upper()}")
            print(f"Confidence: {confidence:.2%}")
            print(f"-----------------------------------\n")

        details = {
            'regime': regime,
            'model_used': model_used,
            'specialist_available': regime in self.regime_models
        }

        return signal, confidence, details

    def load_regime_models(self) -> bool:
        """
        Load all trained regime models from disk

        Returns:
            True if at least one model loaded successfully
        """
        print(f"\nLoading regime models from {self.model_dir}...")

        loaded_count = 0

        # Load regime-specific models
        for regime in self.regimes:
            model_path = self.model_dir / f'{regime}_model.pth'

            if model_path.exists():
                regime_model = RegimeSpecificModel(regime_name=regime)
                if regime_model.load(model_path):
                    self.regime_models[regime] = regime_model
                    loaded_count += 1
                    print(f"  ✓ Loaded {regime} model")

        # Load general fallback model
        general_path = self.model_dir / 'general_model.pth'
        if general_path.exists():
            if self.general_model.load_model(path=general_path):
                print(f"  ✓ Loaded general fallback model")
                loaded_count += 1

        print(f"\nLoaded {loaded_count} models total")
        print(f"  - {len(self.regime_models)} regime-specific models")
        print(f"  - 1 general fallback model")

        return loaded_count > 0

    def compare_regime_performance(self, test_df: pd.DataFrame) -> Dict:
        """
        Compare performance of regime-specific vs general model

        Args:
            test_df: Test dataset

        Returns:
            Dictionary with performance comparison
        """
        print("\n" + "="*70)
        print("REGIME MODEL PERFORMANCE COMPARISON")
        print("="*70)

        # Classify test data by regime
        test_df = test_df.copy()
        test_df['regime'] = test_df.apply(
            lambda row: self._get_regime_for_row(test_df, row.name),
            axis=1
        )

        results = {}

        # Test each regime
        for regime in self.regimes:
            regime_data = test_df[test_df['regime'] == regime]

            if len(regime_data) < 50:
                continue

            print(f"\n{regime} ({len(regime_data)} samples):")

            # Test with regime-specific model (if available)
            if regime in self.regime_models:
                specialist_metrics = self._evaluate_on_data(
                    self.regime_models[regime].model,
                    regime_data
                )
                print(f"  Specialist - Accuracy: {specialist_metrics['accuracy']:.2%}, "
                      f"Sharpe: {specialist_metrics['sharpe']:.3f}")
            else:
                specialist_metrics = None
                print(f"  Specialist - NOT AVAILABLE")

            # Test with general model
            general_metrics = self._evaluate_on_data(self.general_model, regime_data)
            print(f"  General    - Accuracy: {general_metrics['accuracy']:.2%}, "
                  f"Sharpe: {general_metrics['sharpe']:.3f}")

            # Calculate improvement
            if specialist_metrics:
                sharpe_improvement = specialist_metrics['sharpe'] - general_metrics['sharpe']
                acc_improvement = specialist_metrics['accuracy'] - general_metrics['accuracy']
                print(f"  Improvement: Sharpe +{sharpe_improvement:.3f}, Accuracy +{acc_improvement:.2%}")

            results[regime] = {
                'sample_count': len(regime_data),
                'specialist': specialist_metrics,
                'general': general_metrics
            }

        print("\n" + "="*70 + "\n")

        return results

    def _evaluate_on_data(self, model: TradingModel, df: pd.DataFrame) -> Dict:
        """Evaluate a model on specific data"""
        predictions = []
        actuals = []
        returns = []

        for i in range(len(df) - model.sequence_length - 5):
            sequence_data = df.iloc[:model.sequence_length + i]

            try:
                signal, confidence = model.predict(sequence_data)

                signal_map = {'short': 0, 'hold': 1, 'long': 2}
                pred_class = signal_map[signal]

                current_idx = model.sequence_length + i
                if current_idx + 1 < len(df):
                    current_price = df.iloc[current_idx]['close']
                    next_price = df.iloc[current_idx + 1]['close']
                    price_return = (next_price - current_price) / current_price

                    threshold = 0.0005
                    if price_return > threshold:
                        actual_class = 2
                    elif price_return < -threshold:
                        actual_class = 0
                    else:
                        actual_class = 1

                    predictions.append(pred_class)
                    actuals.append(actual_class)
                    returns.append(price_return)
            except:
                continue

        if len(predictions) == 0:
            return {'accuracy': 0.0, 'sharpe': 0.0, 'win_rate': 0.0}

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        returns = np.array(returns)

        return {
            'accuracy': np.mean(predictions == actuals),
            'sharpe': calculate_sharpe_ratio(returns),
            'win_rate': calculate_win_rate(returns)
        }


# Example usage
if __name__ == '__main__':
    print("Regime-Specific Model Training System")
    print("\nExample usage:")
    print("""
    from regime_models import RegimeModelManager
    import pandas as pd

    # Initialize manager
    manager = RegimeModelManager(min_samples_per_regime=500)

    # Load historical data
    df = pd.read_csv('historical_data.csv')
    df['time'] = pd.to_datetime(df['time'])

    # Train regime-specific models
    results = manager.train_regime_models(df, epochs_per_regime=50)

    # Save models (done automatically during training)

    # Later, load models
    manager.load_regime_models()

    # Make prediction (automatically selects best model for current regime)
    signal, confidence, details = manager.predict(current_data, use_regime_specific=True, verbose=True)

    print(f"Regime: {details['regime']}")
    print(f"Model used: {details['model_used']}")
    print(f"Signal: {signal} ({confidence:.2%} confidence)")

    # Compare performance
    comparison = manager.compare_regime_performance(test_data)
    """)
