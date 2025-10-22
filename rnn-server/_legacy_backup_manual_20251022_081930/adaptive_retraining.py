"""
Adaptive Model Retraining System

Automatically monitors model performance and triggers retraining when needed.
Implements walk-forward optimization with scheduled and event-driven retraining.

Expected Impact: +0.2-0.3 Sharpe ratio from staying current with market conditions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import json
from model import TradingModel
from trading_metrics import calculate_sharpe_ratio, calculate_win_rate


class PerformanceMonitor:
    """
    Tracks model performance in real-time and detects degradation
    """

    def __init__(self, window_size: int = 100, alert_threshold: float = 0.15):
        """
        Args:
            window_size: Number of recent predictions to track
            alert_threshold: Percentage drop in performance to trigger alert (0.15 = 15%)
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold

        # Circular buffers for tracking
        self.predictions = []
        self.actuals = []
        self.confidences = []
        self.returns = []
        self.timestamps = []

        # Baseline performance (set during validation)
        self.baseline_accuracy = 0.50
        self.baseline_sharpe = 1.5
        self.baseline_win_rate = 0.52

        # Current performance metrics
        self.current_accuracy = 0.0
        self.current_sharpe = 0.0
        self.current_win_rate = 0.0

    def add_prediction(
        self,
        prediction: str,
        actual_return: float,
        confidence: float,
        timestamp: datetime
    ):
        """
        Add a new prediction and its outcome

        Args:
            prediction: 'long', 'short', or 'hold'
            actual_return: Actual return achieved (positive or negative)
            confidence: Model confidence (0-1)
            timestamp: Time of prediction
        """
        signal_map = {'short': 0, 'hold': 1, 'long': 2}
        pred_class = signal_map[prediction]

        # Determine actual direction
        threshold = 0.0005
        if actual_return > threshold:
            actual_class = 2  # Long
        elif actual_return < -threshold:
            actual_class = 0  # Short
        else:
            actual_class = 1  # Hold

        # Add to buffers (maintain fixed window size)
        self.predictions.append(pred_class)
        self.actuals.append(actual_class)
        self.confidences.append(confidence)
        self.returns.append(actual_return)
        self.timestamps.append(timestamp)

        # Keep only recent window
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.actuals.pop(0)
            self.confidences.pop(0)
            self.returns.pop(0)
            self.timestamps.pop(0)

        # Update current metrics
        self._update_metrics()

    def _update_metrics(self):
        """Calculate current performance metrics"""
        if len(self.predictions) < 20:  # Need minimum data
            return

        preds = np.array(self.predictions)
        actuals = np.array(self.actuals)
        returns = np.array(self.returns)

        # Accuracy
        self.current_accuracy = np.mean(preds == actuals)

        # Win rate (non-hold trades only)
        trade_mask = preds != 1
        if np.sum(trade_mask) > 0:
            trade_returns = returns[trade_mask]
            self.current_win_rate = calculate_win_rate(trade_returns)

        # Sharpe ratio
        if len(returns) > 10:
            self.current_sharpe = calculate_sharpe_ratio(returns)

    def check_degradation(self) -> Dict:
        """
        Check if model performance has degraded significantly

        Returns:
            Dictionary with alert status and metrics
        """
        if len(self.predictions) < self.window_size // 2:
            return {
                'alert': False,
                'reason': 'Insufficient data',
                'metrics': {}
            }

        # Calculate performance drop
        accuracy_drop = (self.baseline_accuracy - self.current_accuracy) / self.baseline_accuracy
        sharpe_drop = (self.baseline_sharpe - self.current_sharpe) / self.baseline_sharpe
        win_rate_drop = (self.baseline_win_rate - self.current_win_rate) / self.baseline_win_rate

        # Check if any metric dropped significantly
        triggers = []
        if accuracy_drop > self.alert_threshold:
            triggers.append(f"Accuracy drop: {accuracy_drop:.1%}")
        if sharpe_drop > self.alert_threshold:
            triggers.append(f"Sharpe drop: {sharpe_drop:.1%}")
        if win_rate_drop > self.alert_threshold:
            triggers.append(f"Win rate drop: {win_rate_drop:.1%}")

        alert = len(triggers) > 0

        return {
            'alert': alert,
            'triggers': triggers,
            'metrics': {
                'current_accuracy': self.current_accuracy,
                'current_sharpe': self.current_sharpe,
                'current_win_rate': self.current_win_rate,
                'baseline_accuracy': self.baseline_accuracy,
                'baseline_sharpe': self.baseline_sharpe,
                'baseline_win_rate': self.baseline_win_rate,
                'accuracy_drop': accuracy_drop,
                'sharpe_drop': sharpe_drop,
                'win_rate_drop': win_rate_drop
            }
        }

    def set_baseline(self, accuracy: float, sharpe: float, win_rate: float):
        """Set baseline performance from validation"""
        self.baseline_accuracy = accuracy
        self.baseline_sharpe = sharpe
        self.baseline_win_rate = win_rate


class AdaptiveRetrainingManager:
    """
    Manages model retraining schedule and triggers
    """

    def __init__(
        self,
        model_dir: str = 'models/adaptive',
        retrain_interval_days: int = 7,
        min_performance_window: int = 100
    ):
        """
        Args:
            model_dir: Directory to store model versions
            retrain_interval_days: Days between scheduled retrains
            min_performance_window: Minimum predictions before checking performance
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.retrain_interval_days = retrain_interval_days
        self.min_performance_window = min_performance_window

        # Performance monitoring
        self.monitor = PerformanceMonitor(window_size=min_performance_window)

        # Retraining state
        self.last_retrain_date = None
        self.retrain_count = 0
        self.state_file = self.model_dir / 'retraining_state.json'

        # Load state if exists
        self._load_state()

    def _load_state(self):
        """Load retraining state from disk"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.last_retrain_date = datetime.fromisoformat(state['last_retrain_date']) if state.get('last_retrain_date') else None
                self.retrain_count = state.get('retrain_count', 0)

                if 'baseline_performance' in state:
                    baseline = state['baseline_performance']
                    self.monitor.set_baseline(
                        baseline['accuracy'],
                        baseline['sharpe'],
                        baseline['win_rate']
                    )

    def _save_state(self):
        """Save retraining state to disk"""
        state = {
            'last_retrain_date': self.last_retrain_date.isoformat() if self.last_retrain_date else None,
            'retrain_count': self.retrain_count,
            'baseline_performance': {
                'accuracy': self.monitor.baseline_accuracy,
                'sharpe': self.monitor.baseline_sharpe,
                'win_rate': self.monitor.baseline_win_rate
            }
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def should_retrain(self, current_time: datetime = None) -> Tuple[bool, str]:
        """
        Check if model should be retrained

        Returns:
            (should_retrain: bool, reason: str)
        """
        if current_time is None:
            current_time = datetime.now()

        # Check scheduled retrain
        if self.last_retrain_date is None:
            return True, "Initial training"

        days_since_retrain = (current_time - self.last_retrain_date).days
        if days_since_retrain >= self.retrain_interval_days:
            return True, f"Scheduled retrain ({days_since_retrain} days since last)"

        # Check performance degradation
        degradation = self.monitor.check_degradation()
        if degradation['alert']:
            reason = f"Performance degradation: {', '.join(degradation['triggers'])}"
            return True, reason

        return False, "No retrain needed"

    def retrain_model(
        self,
        df: pd.DataFrame,
        model: TradingModel,
        epochs: int = 50,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Retrain model with latest data

        Args:
            df: Full historical dataset including recent data
            model: Model instance to retrain
            epochs: Training epochs
            validation_split: Validation data percentage

        Returns:
            Dictionary with retraining results
        """
        print("\n" + "="*70)
        print("ADAPTIVE RETRAINING")
        print("="*70)
        print(f"Retrain #{self.retrain_count + 1}")
        print(f"Last retrain: {self.last_retrain_date}")
        print(f"Data size: {len(df)} bars")

        # Save previous model as backup
        backup_path = self.model_dir / f'model_backup_{self.retrain_count}.pth'
        if model.is_trained:
            model.save_model(path=backup_path)
            print(f"✓ Backed up previous model to {backup_path}")

        # Train model
        print("\nTraining model...")
        model.train(df, epochs=epochs, batch_size=32, validation_split=validation_split)

        # Save new model
        new_model_path = self.model_dir / f'model_v{self.retrain_count + 1}.pth'
        model.save_model(path=new_model_path)

        # Update state
        self.last_retrain_date = datetime.now()
        self.retrain_count += 1

        # Validate on recent data
        validation_size = int(len(df) * validation_split)
        validation_df = df.iloc[-validation_size:]

        validation_metrics = self._validate_model(model, validation_df)

        # Update baseline performance
        self.monitor.set_baseline(
            validation_metrics['accuracy'],
            validation_metrics['sharpe'],
            validation_metrics['win_rate']
        )

        # Reset performance buffer
        self.monitor.predictions.clear()
        self.monitor.actuals.clear()
        self.monitor.returns.clear()

        # Save state
        self._save_state()

        print(f"\n✅ Retraining complete!")
        print(f"New baseline - Accuracy: {validation_metrics['accuracy']:.2%}, "
              f"Sharpe: {validation_metrics['sharpe']:.3f}, "
              f"Win Rate: {validation_metrics['win_rate']:.2%}")
        print("="*70 + "\n")

        return {
            'success': True,
            'retrain_count': self.retrain_count,
            'validation_metrics': validation_metrics,
            'model_path': str(new_model_path)
        }

    def _validate_model(self, model: TradingModel, df: pd.DataFrame) -> Dict:
        """
        Validate model on recent data

        Returns:
            Dictionary with validation metrics
        """
        predictions = []
        actuals = []
        returns = []

        for i in range(len(df) - model.sequence_length - 5):
            sequence_data = df.iloc[:model.sequence_length + i]

            try:
                signal, confidence = model.predict(sequence_data)

                signal_map = {'short': 0, 'hold': 1, 'long': 2}
                pred_class = signal_map[signal]

                # Calculate actual return
                current_idx = model.sequence_length + i
                if current_idx + 1 < len(df):
                    current_price = df.iloc[current_idx]['close']
                    next_price = df.iloc[current_idx + 1]['close']
                    price_return = (next_price - current_price) / current_price

                    # Determine actual label
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

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        returns = np.array(returns)

        accuracy = np.mean(predictions == actuals)
        win_rate = calculate_win_rate(returns)
        sharpe = calculate_sharpe_ratio(returns)

        return {
            'accuracy': accuracy,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'num_predictions': len(predictions)
        }


# Example usage
if __name__ == '__main__':
    print("Adaptive Retraining System")
    print("\nExample usage:")
    print("""
    from adaptive_retraining import AdaptiveRetrainingManager
    import pandas as pd
    from model import TradingModel

    # Initialize manager
    manager = AdaptiveRetrainingManager(
        retrain_interval_days=7,  # Retrain weekly
        min_performance_window=100  # Monitor last 100 predictions
    )

    # Load data
    df = pd.read_csv('historical_data.csv')
    df['time'] = pd.to_datetime(df['time'])

    # Initialize model
    model = TradingModel()

    # Check if retrain needed
    should_retrain, reason = manager.should_retrain()
    if should_retrain:
        print(f"Retraining: {reason}")
        results = manager.retrain_model(df, model)

    # During live trading, track performance
    # After each prediction:
    manager.monitor.add_prediction(
        prediction='long',
        actual_return=0.0012,
        confidence=0.72,
        timestamp=datetime.now()
    )

    # Check degradation
    degradation = manager.monitor.check_degradation()
    if degradation['alert']:
        print(f"Performance degradation detected: {degradation['triggers']}")
    """)
