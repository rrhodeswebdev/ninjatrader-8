"""
Walk-Forward Training Strategy - Complete Implementation

Implements walk-forward optimization to prevent overfitting
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
from advanced_preprocessor import AdvancedDataPreprocessor


class WalkForwardTrainingStrategy:
    """
    Advanced training with walk-forward validation
    """

    def __init__(self, model_class, config: Dict):
        """
        Args:
            model_class: Model class to instantiate
            config: Training configuration
        """
        self.model_class = model_class
        self.config = config
        self.training_stats = {}
        self.preprocessor = AdvancedDataPreprocessor()

    def walk_forward_train(
        self,
        data: pd.DataFrame,
        train_window: int = 252,  # ~1 year of daily data
        test_window: int = 63,    # ~3 months
        step_size: int = 21       # ~1 month
    ) -> Dict:
        """
        Walk-forward optimization

        Args:
            data: Full dataset
            train_window: Training period
            test_window: Testing period
            step_size: Step forward

        Returns:
            Aggregated results
        """
        results = []
        current_pos = 0

        while current_pos + train_window + test_window <= len(data):
            # Split data
            train_data = data.iloc[current_pos:current_pos + train_window]
            test_data = data.iloc[current_pos + train_window:
                                 current_pos + train_window + test_window]

            print(f"\nWalk-Forward Period {len(results) + 1}")
            print(f"Train: {current_pos} to {current_pos + train_window}")
            print(f"Test: {current_pos + train_window} to {current_pos + train_window + test_window}")

            # Train model
            model_snapshot = self._train_period(train_data)

            # Test model
            test_results = self._test_period(model_snapshot, test_data)

            results.append({
                'train_start': current_pos,
                'train_end': current_pos + train_window,
                'test_start': current_pos + train_window,
                'test_end': current_pos + train_window + test_window,
                'metrics': test_results
            })

            # Step forward
            current_pos += step_size

        # Aggregate results
        return self._aggregate_walk_forward_results(results)

    def _train_period(self, train_data: pd.DataFrame) -> nn.Module:
        """
        Train model on single period with early stopping
        """
        # Preprocess data
        train_data_processed = self.preprocessor.preprocess_pipeline(train_data, fit=True)

        # Prepare sequences
        X_train, y_train = self._prepare_sequences(train_data_processed)
        X_val, y_val = self._create_validation_split(X_train, y_train, ratio=0.2)

        # Initialize model
        model = self.model_class(
            input_size=X_train.shape[2],
            **self.config.get('model_params', {})
        )

        # Training loop with early stopping
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 0.001))
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('patience', 10)
        best_model_state = None

        for epoch in range(self.config.get('max_epochs', 100)):
            # Training
            model.train()
            train_loss = self._train_epoch(model, X_train, y_train, optimizer, criterion)

            # Validation
            model.eval()
            val_loss = self._validate_epoch(model, X_val, y_val, criterion)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # Restore best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        return model

    def _test_period(self, model: nn.Module, test_data: pd.DataFrame) -> Dict:
        """Test model on period"""
        # Preprocess
        test_data_processed = self.preprocessor.preprocess_pipeline(test_data, fit=False)

        # Prepare sequences
        X_test, y_test = self._prepare_sequences(test_data_processed)

        # Evaluate
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test)
            y_tensor = torch.LongTensor(y_test)

            logits, _, _ = model(X_tensor)
            predictions = torch.argmax(logits, dim=1)

            accuracy = (predictions == y_tensor).float().mean().item()

        return {'accuracy': accuracy, 'num_samples': len(y_test)}

    def _prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences from dataframe"""
        from advanced_preprocessor import prepare_training_data
        return prepare_training_data(df, sequence_length=self.config.get('sequence_length', 60))

    def _create_validation_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ratio: float = 0.2
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Split into train/val preserving time order"""
        split_idx = int(len(X) * (1 - ratio))
        return (X[:split_idx], y[:split_idx]), (X[split_idx:], y[split_idx:])

    def _train_epoch(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """Train single epoch"""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        optimizer.zero_grad()
        logits, _, _ = model(X_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

        return loss.item()

    def _validate_epoch(self, model: nn.Module, X: np.ndarray, y: np.ndarray, criterion: nn.Module) -> float:
        """Validate single epoch"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y)

            logits, _, _ = model(X_tensor)
            loss = criterion(logits, y_tensor)

        return loss.item()

    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict:
        """Aggregate walk-forward results"""
        accuracies = [r['metrics']['accuracy'] for r in results]

        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'num_periods': len(results),
            'period_results': results
        }
