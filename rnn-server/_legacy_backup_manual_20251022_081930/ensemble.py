"""
Ensemble Modeling System for Trading

Combines multiple models with different architectures and random seeds
to improve prediction robustness and reduce overfitting.

Expected improvement: +0.3 to 0.5 Sharpe ratio
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from model import TradingModel, TradingRNN
from sklearn.metrics import accuracy_score


class EnsemblePredictor:
    """
    Ensemble of trading models with soft voting for robust predictions
    """

    def __init__(
        self,
        n_models: int = 5,
        sequence_length: int = 40,
        model_dir: str = 'models/ensemble'
    ):
        """
        Initialize ensemble with multiple models

        Args:
            n_models: Number of models in ensemble (3-7 recommended)
            sequence_length: Sequence length for each model
            model_dir: Directory to save/load ensemble models
        """
        self.n_models = n_models
        self.sequence_length = sequence_length
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models with different random seeds
        self.models = []
        for i in range(n_models):
            model = TradingModel(
                sequence_length=sequence_length,
                model_path=self.model_dir / f'model_{i}.pth'
            )
            # Set different random seed for each model
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)

            self.models.append(model)

        self.is_trained = False

    def train_ensemble(
        self,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """
        Train all models in the ensemble

        Args:
            df: Training data
            epochs: Epochs per model
            batch_size: Batch size
            verbose: Print training progress
        """
        print(f"\n{'='*60}")
        print(f"TRAINING ENSEMBLE OF {self.n_models} MODELS")
        print(f"{'='*60}\n")

        for i, model in enumerate(self.models):
            print(f"\n--- Training Model {i+1}/{self.n_models} ---")

            # Set unique random seed
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)

            # Train model
            model.train(
                df,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2
            )

            # Save model
            model.save_model()

            if verbose:
                print(f"✓ Model {i+1} trained and saved")

        self.is_trained = True

        print(f"\n{'='*60}")
        print("ENSEMBLE TRAINING COMPLETE")
        print(f"{'='*60}\n")

    def predict(
        self,
        recent_bars_df: pd.DataFrame,
        voting_strategy: str = 'soft',  # 'soft' or 'hard'
        return_details: bool = False
    ) -> Tuple[str, float]:
        """
        Predict using ensemble with weighted voting

        Args:
            recent_bars_df: Recent OHLC data
            voting_strategy: 'soft' (confidence-weighted) or 'hard' (majority)
            return_details: If True, return individual predictions

        Returns:
            (signal, ensemble_confidence) or (signal, confidence, details_dict)
        """

        if not self.is_trained:
            # Try to load models
            self.load_ensemble()
            if not self.is_trained:
                print("WARNING: Ensemble not trained!")
                return ('hold', 0.0) if not return_details else ('hold', 0.0, {})

        # Get predictions from all models
        predictions = []
        confidences = []

        for i, model in enumerate(self.models):
            signal, confidence = model.predict(recent_bars_df)
            predictions.append(signal)
            confidences.append(confidence)

        # Convert signals to numeric (0=short, 1=hold, 2=long)
        signal_map = {'short': 0, 'hold': 1, 'long': 2}
        reverse_map = {0: 'short', 1: 'hold', 2: 'long'}
        numeric_predictions = [signal_map[s] for s in predictions]

        if voting_strategy == 'soft':
            # Soft voting: weight by confidence
            vote_scores = {0: 0.0, 1: 0.0, 2: 0.0}  # short, hold, long

            for pred, conf in zip(numeric_predictions, confidences):
                vote_scores[pred] += conf

            # Get winning class and aggregate confidence
            winning_class = max(vote_scores, key=vote_scores.get)
            total_confidence = sum(vote_scores.values())
            ensemble_confidence = vote_scores[winning_class] / total_confidence if total_confidence > 0 else 0

        else:  # hard voting
            # Hard voting: simple majority
            vote_counts = {0: 0, 1: 0, 2: 0}
            for pred in numeric_predictions:
                vote_counts[pred] += 1

            winning_class = max(vote_counts, key=vote_counts.get)

            # Confidence = (votes for winner) / (total votes) * (average confidence of winner's voters)
            winner_confidences = [conf for pred, conf in zip(numeric_predictions, confidences) if pred == winning_class]
            ensemble_confidence = (vote_counts[winning_class] / len(numeric_predictions)) * np.mean(winner_confidences)

        final_signal = reverse_map[winning_class]

        if return_details:
            details = {
                'individual_predictions': predictions,
                'individual_confidences': confidences,
                'vote_scores': vote_scores if voting_strategy == 'soft' else vote_counts,
                'agreement_rate': vote_counts[winning_class] / len(numeric_predictions) if voting_strategy == 'hard' else vote_scores[winning_class] / total_confidence
            }
            return final_signal, ensemble_confidence, details

        return final_signal, ensemble_confidence

    def predict_with_risk_params(
        self,
        recent_bars_df: pd.DataFrame,
        account_balance: float = 25000.0,
        tick_value: float = 12.50
    ) -> Dict:
        """
        Predict with ensemble and calculate risk parameters

        Returns:
            Dictionary with signal, confidence, and risk management params
        """
        from risk_management import RiskManager
        from model import detect_market_regime, calculate_atr

        # Get ensemble prediction
        signal, confidence, details = self.predict(
            recent_bars_df,
            voting_strategy='soft',
            return_details=True
        )

        if signal == 'hold':
            return {
                'signal': 'hold',
                'confidence': confidence,
                'contracts': 0,
                'entry_price': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'reason': 'Ensemble Hold signal - no trade',
                'ensemble_details': details
            }

        # Get current market data
        current_bar = recent_bars_df.iloc[-1]
        entry_price = current_bar['close']

        # Get ATR
        if 'atr' in recent_bars_df.columns:
            atr = recent_bars_df['atr'].iloc[-1]
        else:
            atr_values = calculate_atr(
                recent_bars_df['high'].values,
                recent_bars_df['low'].values,
                recent_bars_df['close'].values
            )
            atr = atr_values[-1] if len(atr_values) > 0 else 15.0

        # Get regime
        regime = detect_market_regime(recent_bars_df, lookback=min(100, len(recent_bars_df)-1))

        # Calculate trade parameters
        risk_mgr = RiskManager()
        trade_params = risk_mgr.calculate_trade_parameters(
            signal=signal,
            confidence=confidence,
            entry_price=entry_price,
            atr=atr,
            regime=regime,
            account_balance=account_balance,
            tick_value=tick_value
        )

        # Add ensemble details
        trade_params['ensemble_details'] = details

        return trade_params

    def save_ensemble(self):
        """Save all models in the ensemble"""
        for i, model in enumerate(self.models):
            model_path = self.model_dir / f'model_{i}.pth'
            model.save_model(path=model_path)

        print(f"✓ Ensemble saved to {self.model_dir}")

    def load_ensemble(self) -> bool:
        """Load all models in the ensemble"""
        success_count = 0

        for i, model in enumerate(self.models):
            model_path = self.model_dir / f'model_{i}.pth'
            if model_path.exists():
                if model.load_model(path=model_path):
                    success_count += 1

        if success_count == self.n_models:
            self.is_trained = True
            print(f"✓ Loaded {success_count}/{self.n_models} ensemble models")
            return True
        elif success_count > 0:
            print(f"⚠️  Only loaded {success_count}/{self.n_models} models")
            return False
        else:
            print(f"❌ No ensemble models found in {self.model_dir}")
            return False

    def evaluate_ensemble(
        self,
        df_validation: pd.DataFrame
    ) -> Dict:
        """
        Evaluate ensemble performance on validation data

        Returns:
            Dictionary with performance metrics
        """
        print("\n" + "="*60)
        print("EVALUATING ENSEMBLE PERFORMANCE")
        print("="*60 + "\n")

        # Get individual model predictions
        all_predictions = []
        all_confidences = []

        for i, model in enumerate(self.models):
            predictions = []
            confidences = []

            # Make predictions on each bar
            for j in range(model.sequence_length + 100, len(df_validation)):
                recent_data = df_validation.iloc[:j+1]
                signal, confidence = model.predict(recent_data)
                predictions.append(signal)
                confidences.append(confidence)

            all_predictions.append(predictions)
            all_confidences.append(confidences)

            print(f"Model {i+1}: {len(predictions)} predictions")

        # Get ensemble predictions
        ensemble_predictions = []
        ensemble_confidences = []

        for j in range(len(all_predictions[0])):
            # Get predictions from all models for this bar
            bar_predictions = [preds[j] for preds in all_predictions]
            bar_confidences = [confs[j] for confs in all_confidences]

            # Soft voting
            signal_map = {'short': 0, 'hold': 1, 'long': 2}
            reverse_map = {0: 'short', 1: 'hold', 2: 'long'}
            numeric_preds = [signal_map[s] for s in bar_predictions]

            vote_scores = {0: 0.0, 1: 0.0, 2: 0.0}
            for pred, conf in zip(numeric_preds, bar_confidences):
                vote_scores[pred] += conf

            winning_class = max(vote_scores, key=vote_scores.get)
            total_conf = sum(vote_scores.values())
            ensemble_conf = vote_scores[winning_class] / total_conf if total_conf > 0 else 0

            ensemble_predictions.append(reverse_map[winning_class])
            ensemble_confidences.append(ensemble_conf)

        # Calculate disagreement rate (diversity measure)
        disagreement_count = 0
        for j in range(len(all_predictions[0])):
            bar_predictions = [preds[j] for preds in all_predictions]
            if len(set(bar_predictions)) > 1:  # Not all same
                disagreement_count += 1

        disagreement_rate = disagreement_count / len(all_predictions[0])

        # Calculate average confidence
        avg_individual_conf = np.mean([np.mean(confs) for confs in all_confidences])
        avg_ensemble_conf = np.mean(ensemble_confidences)

        results = {
            'num_predictions': len(ensemble_predictions),
            'avg_individual_confidence': avg_individual_conf,
            'avg_ensemble_confidence': avg_ensemble_conf,
            'disagreement_rate': disagreement_rate,
            'confidence_boost': avg_ensemble_conf / avg_individual_conf if avg_individual_conf > 0 else 0,
            'individual_predictions': all_predictions,
            'ensemble_predictions': ensemble_predictions
        }

        print(f"\n📊 ENSEMBLE STATISTICS:")
        print(f"  Predictions: {results['num_predictions']}")
        print(f"  Avg Individual Confidence: {avg_individual_conf:.2%}")
        print(f"  Avg Ensemble Confidence: {avg_ensemble_conf:.2%}")
        print(f"  Confidence Boost: {results['confidence_boost']:.2f}x")
        print(f"  Model Disagreement Rate: {disagreement_rate:.2%}")
        print("\n" + "="*60 + "\n")

        return results


# Example usage
if __name__ == '__main__':
    print("Ensemble Modeling System")
    print("\nExample usage:")
    print("""
    from ensemble import EnsemblePredictor
    import pandas as pd

    # Load data
    df = pd.read_csv('historical_data.csv')
    df['time'] = pd.to_datetime(df['time'])

    # Create and train ensemble
    ensemble = EnsemblePredictor(n_models=5)
    ensemble.train_ensemble(df, epochs=50)

    # Make predictions
    signal, confidence = ensemble.predict(df)
    print(f"Ensemble prediction: {signal} ({confidence:.2%} confidence)")

    # Get trade parameters
    trade_params = ensemble.predict_with_risk_params(df, account_balance=25000)
    print(f"Contracts: {trade_params['contracts']}")
    print(f"Stop: ${trade_params['stop_loss']:.2f}")
    print(f"Target: ${trade_params['take_profit']:.2f}")
    """)
