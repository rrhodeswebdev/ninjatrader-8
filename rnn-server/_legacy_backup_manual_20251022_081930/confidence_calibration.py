"""
Confidence Calibration System
==============================

Tracks and evaluates how well model confidence scores match actual win rates.
A well-calibrated model should have:
- 70% confidence → ~70% win rate
- 80% confidence → ~80% win rate

If poorly calibrated, the model's confidence scores are not trustworthy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path


class ConfidenceCalibrator:
    """
    Track and analyze confidence calibration
    """

    def __init__(self, n_bins: int = 10):
        """
        Args:
            n_bins: Number of confidence bins (e.g., 10 = 0-0.1, 0.1-0.2, ...)
        """
        self.n_bins = n_bins
        self.predictions = []  # List of (confidence, was_correct) tuples

    def add_prediction(self, confidence: float, was_correct: bool):
        """
        Add a prediction outcome

        Args:
            confidence: Model confidence (0-1)
            was_correct: Whether prediction was correct
        """
        self.predictions.append((confidence, was_correct))

    def calculate_calibration(self) -> Dict:
        """
        Calculate calibration metrics

        Returns:
            Dictionary with calibration analysis
        """
        if len(self.predictions) < 10:
            return {
                'error': 'Not enough predictions for calibration analysis',
                'num_predictions': len(self.predictions)
            }

        # Convert to arrays
        confidences = np.array([p[0] for p in self.predictions])
        corrects = np.array([p[1] for p in self.predictions])

        # Create bins
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        # Calculate accuracy per bin
        bin_data = []
        for i in range(self.n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_conf = confidences[mask]
                bin_correct = corrects[mask]

                bin_data.append({
                    'bin_idx': i,
                    'bin_range': (bins[i], bins[i+1]),
                    'avg_confidence': np.mean(bin_conf),
                    'actual_accuracy': np.mean(bin_correct),
                    'count': len(bin_conf),
                    'gap': np.mean(bin_conf) - np.mean(bin_correct)  # Calibration error
                })

        # Calculate Expected Calibration Error (ECE)
        ece = 0.0
        total_count = len(confidences)
        for bin_info in bin_data:
            bin_weight = bin_info['count'] / total_count
            ece += bin_weight * abs(bin_info['gap'])

        # Calculate Maximum Calibration Error (MCE)
        mce = max([abs(bin_info['gap']) for bin_info in bin_data]) if bin_data else 0.0

        # Overall metrics
        overall_confidence = np.mean(confidences)
        overall_accuracy = np.mean(corrects)
        overall_gap = overall_confidence - overall_accuracy

        results = {
            'num_predictions': len(self.predictions),
            'bin_data': bin_data,
            'expected_calibration_error': ece,
            'max_calibration_error': mce,
            'overall_confidence': overall_confidence,
            'overall_accuracy': overall_accuracy,
            'overall_gap': overall_gap,
            'is_overconfident': overall_gap > 0.05,  # Model thinks it's better than it is
            'is_underconfident': overall_gap < -0.05,  # Model is better than it thinks
            'is_well_calibrated': abs(overall_gap) < 0.05 and ece < 0.10
        }

        return results

    def plot_calibration_curve(self, save_path: str = None):
        """
        Plot calibration curve

        Args:
            save_path: Optional path to save plot
        """
        results = self.calculate_calibration()

        if 'error' in results:
            print(f"Cannot plot: {results['error']}")
            return

        bin_data = results['bin_data']

        # Extract data for plotting
        avg_confidences = [b['avg_confidence'] for b in bin_data]
        actual_accuracies = [b['actual_accuracy'] for b in bin_data]
        counts = [b['count'] for b in bin_data]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Calibration Curve
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        ax1.scatter(avg_confidences, actual_accuracies, s=np.array(counts)*2, alpha=0.6, c='blue')
        ax1.plot(avg_confidences, actual_accuracies, 'b-', alpha=0.5)

        ax1.set_xlabel('Mean Predicted Confidence', fontsize=12)
        ax1.set_ylabel('Actual Accuracy', fontsize=12)
        ax1.set_title(f'Calibration Curve (ECE: {results["expected_calibration_error"]:.3f})', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

        # Plot 2: Bin Histogram
        bin_ranges = [f"{b['bin_range'][0]:.1f}-{b['bin_range'][1]:.1f}" for b in bin_data]
        x_pos = np.arange(len(bin_ranges))

        ax2.bar(x_pos, counts, alpha=0.7, color='blue')
        ax2.set_xlabel('Confidence Bin', fontsize=12)
        ax2.set_ylabel('Number of Predictions', fontsize=12)
        ax2.set_title('Prediction Distribution by Confidence', fontsize=14)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(bin_ranges, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Calibration curve saved to {save_path}")
        else:
            plt.show()

    def print_calibration_report(self):
        """Print detailed calibration report"""
        results = self.calculate_calibration()

        if 'error' in results:
            print(f"❌ {results['error']}")
            return

        print("\n" + "="*70)
        print("CONFIDENCE CALIBRATION REPORT")
        print("="*70)

        print(f"\n📊 OVERALL METRICS:")
        print(f"  Total Predictions: {results['num_predictions']}")
        print(f"  Average Confidence: {results['overall_confidence']:.2%}")
        print(f"  Actual Accuracy: {results['overall_accuracy']:.2%}")
        print(f"  Confidence Gap: {results['overall_gap']:+.2%}")

        print(f"\n📏 CALIBRATION ERRORS:")
        print(f"  Expected Calibration Error (ECE): {results['expected_calibration_error']:.3f}")
        print(f"  Maximum Calibration Error (MCE): {results['max_calibration_error']:.3f}")

        print(f"\n🎯 CALIBRATION STATUS:")
        if results['is_well_calibrated']:
            print("  ✅ Model is WELL CALIBRATED (ECE < 0.10, |gap| < 0.05)")
        elif results['is_overconfident']:
            print("  ⚠️  Model is OVERCONFIDENT (predicts higher confidence than actual)")
            print("     → Consider: Temperature scaling, more dropout, or simpler model")
        elif results['is_underconfident']:
            print("  ⚠️  Model is UNDERCONFIDENT (actual performance better than confidence)")
            print("     → Consider: Reduce regularization or use confidence boosting")
        else:
            print("  ⚠️  Model calibration needs improvement")

        print(f"\n📋 BIN-BY-BIN ANALYSIS:")
        print(f"  {'Bin Range':<15} {'Count':<8} {'Avg Conf':<12} {'Accuracy':<12} {'Gap':<12}")
        print("  " + "-"*65)

        for bin_info in results['bin_data']:
            bin_range = f"{bin_info['bin_range'][0]:.1f}-{bin_info['bin_range'][1]:.1f}"
            count = bin_info['count']
            avg_conf = bin_info['avg_confidence']
            accuracy = bin_info['actual_accuracy']
            gap = bin_info['gap']

            gap_indicator = "✅" if abs(gap) < 0.05 else "⚠️ "

            print(f"  {bin_range:<15} {count:<8} {avg_conf:<12.2%} {accuracy:<12.2%} {gap:+.2%} {gap_indicator}")

        print("="*70 + "\n")


class TemperatureScaler:
    """
    Apply temperature scaling to calibrate confidence scores

    Temperature scaling: p_calibrated = softmax(logits / T)
    - T > 1: Reduce confidence (smooth distribution)
    - T < 1: Increase confidence (sharpen distribution)
    - T = 1: No change
    """

    def __init__(self):
        self.temperature = 1.0

    def find_temperature(
        self,
        confidences: np.ndarray,
        correct_labels: np.ndarray,
        t_range: Tuple[float, float] = (0.1, 5.0),
        n_trials: int = 50
    ) -> float:
        """
        Find optimal temperature using grid search

        Args:
            confidences: Array of confidence scores
            correct_labels: Array of boolean (was prediction correct?)
            t_range: Range of temperatures to try
            n_trials: Number of temperature values to try

        Returns:
            Optimal temperature
        """
        temperatures = np.linspace(t_range[0], t_range[1], n_trials)
        best_ece = float('inf')
        best_t = 1.0

        for t in temperatures:
            # Apply temperature scaling (simplified)
            scaled_conf = self._apply_temperature(confidences, t)

            # Calculate ECE
            ece = self._calculate_ece(scaled_conf, correct_labels)

            if ece < best_ece:
                best_ece = ece
                best_t = t

        self.temperature = best_t
        print(f"✅ Optimal temperature found: {best_t:.3f} (ECE: {best_ece:.4f})")
        return best_t

    def _apply_temperature(self, confidences: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling (simplified version)"""
        # Convert confidence to pseudo-logit, scale, convert back
        epsilon = 1e-7
        confidences = np.clip(confidences, epsilon, 1 - epsilon)
        logits = np.log(confidences / (1 - confidences))
        scaled_logits = logits / temperature
        scaled_conf = 1 / (1 + np.exp(-scaled_logits))
        return scaled_conf

    def _calculate_ece(self, confidences: np.ndarray, corrects: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_conf = confidences[mask]
                bin_correct = corrects[mask]
                bin_weight = len(bin_conf) / len(confidences)
                gap = abs(np.mean(bin_conf) - np.mean(bin_correct))
                ece += bin_weight * gap

        return ece

    def scale(self, confidence: float) -> float:
        """Apply learned temperature scaling to a single confidence score"""
        return float(self._apply_temperature(np.array([confidence]), self.temperature)[0])


# Example usage
if __name__ == '__main__':
    print("Confidence Calibration System")
    print("="*50)
    print("\nExample usage:")
    print("""
    from confidence_calibration import ConfidenceCalibrator

    # Initialize calibrator
    calibrator = ConfidenceCalibrator(n_bins=10)

    # During validation/testing, track predictions
    for prediction in validation_predictions:
        signal, confidence = model.predict(data)
        was_correct = (signal == actual_outcome)
        calibrator.add_prediction(confidence, was_correct)

    # Analyze calibration
    calibrator.print_calibration_report()
    calibrator.plot_calibration_curve('models/calibration_curve.png')

    # If poorly calibrated, apply temperature scaling
    if not calibrator.calculate_calibration()['is_well_calibrated']:
        from confidence_calibration import TemperatureScaler
        scaler = TemperatureScaler()
        scaler.find_temperature(confidences, corrects)

        # Use scaled confidence in production
        raw_conf = 0.85
        calibrated_conf = scaler.scale(raw_conf)
        print(f"Raw: {raw_conf:.2%} → Calibrated: {calibrated_conf:.2%}")
    """)
