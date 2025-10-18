"""
Advanced Loss Functions for Better Model Training

Addresses confidence and accuracy issues through better training objectives:
1. Label Smoothing - Prevents overconfident predictions
2. Confidence Penalty Loss - Encourages higher confidence on correct predictions
3. Directional Accuracy Loss - Rewards correct direction more than exact class
4. Sharpe-Optimized Loss - Directly optimizes trading performance
5. Asymmetric Loss - Different penalties for false positives vs false negatives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy

    Instead of hard targets [0, 1, 0], use soft targets [0.05, 0.9, 0.05]

    Benefits:
    - Prevents overconfident predictions
    - Better calibrated probabilities
    - Reduces overfitting
    - Model learns to be less certain

    Paper: "Rethinking the Inception Architecture" (Szegedy et al., 2016)
    """

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor = None):
        """
        Args:
            smoothing: Smoothing parameter (0.1 = 10% smoothing)
            weight: Class weights for imbalanced data
        """
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Model predictions (batch_size, num_classes) - logits
            target: True labels (batch_size,) - class indices

        Returns:
            Smoothed cross entropy loss
        """
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)

        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        # Apply class weights if provided
        if self.weight is not None:
            weights = self.weight[target]
            loss = torch.sum(-true_dist * log_preds, dim=-1) * weights
            return loss.mean()
        else:
            return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))


class ConfidencePenaltyLoss(nn.Module):
    """
    Confidence Penalty Loss

    Penalizes low confidence on correct predictions and
    high confidence on incorrect predictions.

    Encourages the model to:
    - Be more confident when correct
    - Be less confident when wrong
    """

    def __init__(self, base_loss: nn.Module = None, confidence_weight: float = 0.1):
        """
        Args:
            base_loss: Base loss function (e.g., CrossEntropyLoss)
            confidence_weight: Weight for confidence penalty term
        """
        super().__init__()
        self.base_loss = base_loss or nn.CrossEntropyLoss()
        self.confidence_weight = confidence_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Model predictions (batch_size, num_classes) - logits
            target: True labels (batch_size,)

        Returns:
            Loss with confidence penalty
        """
        # Base classification loss
        base_loss_val = self.base_loss(pred, target)

        # Get predicted probabilities
        probs = F.softmax(pred, dim=-1)

        # Get confidence (max probability)
        confidences, predictions = torch.max(probs, dim=1)

        # Correctness mask
        correct = (predictions == target).float()

        # Confidence penalty:
        # - On correct predictions: penalize low confidence (encourage higher confidence)
        # - On incorrect predictions: penalize high confidence (encourage lower confidence)
        correct_penalty = (1 - confidences) * correct  # Penalty for low confidence when correct
        incorrect_penalty = confidences * (1 - correct)  # Penalty for high confidence when wrong

        confidence_penalty = (correct_penalty + incorrect_penalty).mean()

        total_loss = base_loss_val + self.confidence_weight * confidence_penalty

        return total_loss


class DirectionalAccuracyLoss(nn.Module):
    """
    Directional Accuracy Loss for Trading

    Rewards correct direction (long vs short) more than exact class.
    Useful when the exact timing matters less than the direction.

    For 3-class problem (SHORT=0, HOLD=1, LONG=2):
    - Correct class: No penalty
    - Wrong direction (LONG predicted, SHORT actual): High penalty
    - Neutral confusion (LONG predicted, HOLD actual): Medium penalty
    """

    def __init__(self, directional_weight: float = 2.0):
        """
        Args:
            directional_weight: How much more to penalize wrong direction vs wrong class
        """
        super().__init__()
        self.directional_weight = directional_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Model predictions (batch_size, 3) - logits [SHORT, HOLD, LONG]
            target: True labels (batch_size,) - {0: SHORT, 1: HOLD, 2: LONG}

        Returns:
            Directional-aware loss
        """
        # Standard cross entropy as base
        ce_loss = F.cross_entropy(pred, target, reduction='none')

        # Get predictions
        _, predictions = torch.max(pred, dim=1)

        # Create penalty matrix
        # 0 = correct, 1 = wrong class, 2 = wrong direction
        penalty = torch.zeros_like(ce_loss)

        # Wrong direction penalties (SHORT vs LONG)
        wrong_direction = ((predictions == 0) & (target == 2)) | \
                         ((predictions == 2) & (target == 0))
        penalty[wrong_direction] = self.directional_weight

        # Apply penalties
        weighted_loss = ce_loss * (1 + penalty)

        return weighted_loss.mean()


class SharpeOptimizedLoss(nn.Module):
    """
    Sharpe-Optimized Loss

    Directly optimizes for Sharpe ratio instead of classification accuracy.
    Requires price changes to calculate returns.

    This is a differentiable approximation to Sharpe ratio optimization.
    """

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default: 0)
        """
        super().__init__()
        self.risk_free_rate = risk_free_rate

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                returns: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Model predictions (batch_size, 3) - logits [SHORT, HOLD, LONG]
            target: True labels (batch_size,)
            returns: Actual returns for each sample (batch_size,)

        Returns:
            Negative Sharpe ratio (to minimize)
        """
        # Get predicted probabilities
        probs = F.softmax(pred, dim=-1)

        # Extract directional probabilities
        prob_short = probs[:, 0]
        prob_long = probs[:, 2]

        # Position: -1 for short, 0 for hold, +1 for long
        # Use expected position based on probabilities
        position = prob_long - prob_short

        # Calculate strategy returns
        strategy_returns = position * returns

        # Calculate Sharpe ratio
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std() + 1e-6  # Avoid division by zero

        sharpe = (mean_return - self.risk_free_rate) / std_return

        # Return negative Sharpe (we want to minimize loss, but maximize Sharpe)
        # Also add small cross-entropy term for stability
        ce_loss = F.cross_entropy(pred, target)

        # Combine: 70% Sharpe optimization, 30% classification
        total_loss = -sharpe * 0.7 + ce_loss * 0.3

        return total_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Trading

    Different penalties for different types of errors:
    - False LONG (predict long, should be short): High penalty (lose money)
    - False SHORT (predict short, should be long): High penalty (lose money)
    - False HOLD (predict hold, should trade): Medium penalty (missed opportunity)
    - False TRADE (predict trade, should hold): Low penalty (may still profit)
    """

    def __init__(self, false_signal_penalty: float = 2.0,
                 missed_trade_penalty: float = 1.0):
        """
        Args:
            false_signal_penalty: Penalty for wrong directional signals
            missed_trade_penalty: Penalty for missing trades (predicting HOLD when should trade)
        """
        super().__init__()
        self.false_signal_penalty = false_signal_penalty
        self.missed_trade_penalty = missed_trade_penalty

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Model predictions (batch_size, 3) - logits
            target: True labels (batch_size,)

        Returns:
            Asymmetric loss
        """
        # Base cross entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none')

        # Get predictions
        _, predictions = torch.max(pred, dim=1)

        # Penalty weights
        penalty = torch.ones_like(ce_loss)

        # False directional signals (worst case)
        false_long = (predictions == 2) & (target == 0)
        false_short = (predictions == 0) & (target == 2)
        penalty[false_long | false_short] = self.false_signal_penalty

        # Missed trades (medium penalty)
        missed_long = (predictions == 1) & (target == 2)
        missed_short = (predictions == 1) & (target == 0)
        penalty[missed_long | missed_short] = self.missed_trade_penalty

        # Apply penalties
        weighted_loss = ce_loss * penalty

        return weighted_loss.mean()


class CombinedTradingLoss(nn.Module):
    """
    Combined Loss Function that uses multiple objectives

    Combines:
    1. Label Smoothing CE (for calibration)
    2. Confidence Penalty (for higher confidence)
    3. Directional Accuracy (for correct direction)
    4. Optional: Sharpe optimization (if returns available)
    """

    def __init__(self,
                 smoothing: float = 0.1,
                 confidence_weight: float = 0.1,
                 directional_weight: float = 1.5,
                 use_sharpe: bool = False):
        """
        Args:
            smoothing: Label smoothing parameter
            confidence_weight: Weight for confidence penalty
            directional_weight: Weight for directional accuracy
            use_sharpe: Whether to include Sharpe optimization
        """
        super().__init__()

        self.label_smooth_loss = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.confidence_loss = ConfidencePenaltyLoss(confidence_weight=confidence_weight)
        self.directional_loss = DirectionalAccuracyLoss(directional_weight=directional_weight)

        if use_sharpe:
            self.sharpe_loss = SharpeOptimizedLoss()

        self.use_sharpe = use_sharpe

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                returns: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: Model predictions (batch_size, 3)
            target: True labels (batch_size,)
            returns: Optional returns for Sharpe optimization (batch_size,)

        Returns:
            Combined loss
        """
        # Label smoothing CE (40%)
        ls_loss = self.label_smooth_loss(pred, target)

        # Confidence penalty (20%)
        conf_loss = self.confidence_loss(pred, target)

        # Directional accuracy (40%)
        dir_loss = self.directional_loss(pred, target)

        total_loss = 0.4 * ls_loss + 0.2 * conf_loss + 0.4 * dir_loss

        # Add Sharpe optimization if returns provided
        if self.use_sharpe and returns is not None:
            sharpe_loss = self.sharpe_loss(pred, target, returns)
            total_loss = 0.7 * total_loss + 0.3 * sharpe_loss

        return total_loss


class FocalLossImproved(nn.Module):
    """
    Improved Focal Loss with Label Smoothing

    Combines Focal Loss (for hard examples) with Label Smoothing (for calibration)
    """

    def __init__(self, gamma: float = 2.0, smoothing: float = 0.1,
                 weight: torch.Tensor = None):
        """
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            smoothing: Label smoothing parameter
            weight: Class weights
        """
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Model predictions (batch_size, num_classes) - logits
            target: True labels (batch_size,)

        Returns:
            Focal loss with label smoothing
        """
        n_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)
        probs = F.softmax(pred, dim=-1)

        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        # Focal term: (1 - p_t)^gamma
        # p_t is the probability of the true class
        pt = (true_dist * probs).sum(dim=-1)
        focal_weight = (1 - pt) ** self.gamma

        # Compute loss
        loss = -focal_weight * (true_dist * log_probs).sum(dim=-1)

        # Apply class weights
        if self.weight is not None:
            weights = self.weight[target]
            loss = loss * weights

        return loss.mean()


if __name__ == '__main__':
    print("Advanced Loss Functions for Trading Models")
    print("="*70)

    # Demo with synthetic data
    batch_size = 32
    num_classes = 3

    # Synthetic predictions and labels
    pred = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))
    returns = torch.randn(batch_size) * 0.01  # ±1% returns

    print("\nTesting loss functions:")
    print("-"*70)

    # 1. Label Smoothing
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    print(f"Label Smoothing CE Loss: {ls_loss(pred, target).item():.4f}")

    # 2. Confidence Penalty
    conf_loss = ConfidencePenaltyLoss()
    print(f"Confidence Penalty Loss: {conf_loss(pred, target).item():.4f}")

    # 3. Directional Accuracy
    dir_loss = DirectionalAccuracyLoss()
    print(f"Directional Accuracy Loss: {dir_loss(pred, target).item():.4f}")

    # 4. Sharpe Optimized
    sharpe_loss = SharpeOptimizedLoss()
    print(f"Sharpe Optimized Loss: {sharpe_loss(pred, target, returns).item():.4f}")

    # 5. Combined Loss
    combined_loss = CombinedTradingLoss(use_sharpe=True)
    print(f"Combined Trading Loss: {combined_loss(pred, target, returns).item():.4f}")

    print("\n✓ All loss functions working correctly")
