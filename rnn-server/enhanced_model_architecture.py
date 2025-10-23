"""
Enhanced Trading Model Architecture - Complete Implementation

Implements the advanced LSTM architecture from the quant analysis:
- Multi-scale LSTM branches (short, medium, long term)
- Multi-head attention mechanism
- Feature fusion layers
- Uncertainty quantification via MC Dropout
- Residual connections for better gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class EnhancedTradingModel(nn.Module):
    """
    Advanced LSTM with attention and multi-scale processing
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,  # long, short, no_trade
        sequence_length: int = 60
    ):
        """
        Initialize enhanced model

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            num_classes: Number of output classes
            sequence_length: Input sequence length
        """
        super(EnhancedTradingModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        # Multi-scale LSTM branches
        self.lstm_short = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        self.lstm_medium = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        self.lstm_long = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        # Batch normalization for each LSTM output
        self.bn_short = nn.BatchNorm1d(hidden_size)
        self.bn_medium = nn.BatchNorm1d(hidden_size)
        self.bn_long = nn.BatchNorm1d(hidden_size)

        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 3,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Feature fusion network
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Residual connection
        self.residual_projection = nn.Linear(hidden_size * 3, hidden_size)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # Uncertainty estimation head (Monte Carlo Dropout)
        self.uncertainty_dropout = nn.Dropout(0.5)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        mc_samples: int = 1
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with multi-scale processing and attention

        Args:
            x: Input tensor [batch, sequence, features]
            return_attention: Return attention weights
            mc_samples: Number of MC dropout samples for uncertainty

        Returns:
            Tuple of (logits, uncertainty, attention_weights)
            - logits: Class probabilities
            - uncertainty: Predictive uncertainty (optional)
            - attention_weights: Attention weights (optional)
        """
        batch_size = x.size(0)

        # Determine sequence lengths for multi-scale processing
        seq_len = x.size(1)
        short_len = min(10, seq_len)
        medium_len = min(30, seq_len)
        long_len = seq_len

        # Process at multiple time scales
        # Short-term: last 10 bars
        x_short = x[:, -short_len:, :]
        lstm_short_out, _ = self.lstm_short(x_short)
        short_features = lstm_short_out[:, -1, :]
        short_features = self.bn_short(short_features)

        # Medium-term: last 30 bars
        x_medium = x[:, -medium_len:, :]
        lstm_medium_out, _ = self.lstm_medium(x_medium)
        medium_features = lstm_medium_out[:, -1, :]
        medium_features = self.bn_medium(medium_features)

        # Long-term: all bars
        lstm_long_out, _ = self.lstm_long(x)
        long_features = lstm_long_out[:, -1, :]
        long_features = self.bn_long(long_features)

        # Concatenate multi-scale features
        combined = torch.cat([short_features, medium_features, long_features], dim=1)

        # Apply attention
        combined_for_attention = combined.unsqueeze(1)  # Add sequence dimension
        attended, attention_weights = self.attention(
            combined_for_attention,
            combined_for_attention,
            combined_for_attention
        )
        attended = attended.squeeze(1)

        # Feature fusion with residual connection
        fused = self.fusion(attended)

        # Residual connection
        residual = self.residual_projection(combined)
        fused = fused + residual

        # Classification
        logits = self.classifier(fused)

        # Uncertainty estimation via MC Dropout
        if mc_samples > 1:
            uncertainty = self._estimate_uncertainty(fused, mc_samples)
        else:
            uncertainty = None

        if return_attention:
            return logits, uncertainty, attention_weights
        else:
            return logits, uncertainty, None

    def _estimate_uncertainty(self, features: torch.Tensor, mc_samples: int) -> torch.Tensor:
        """
        Estimate predictive uncertainty using Monte Carlo Dropout

        Returns epistemic uncertainty (model uncertainty)

        Args:
            features: Feature tensor
            mc_samples: Number of MC samples

        Returns:
            Uncertainty scores [batch]
        """
        self.train()  # Enable dropout

        predictions = []
        for _ in range(mc_samples):
            dropped_features = self.uncertainty_dropout(features)
            logits = self.classifier(dropped_features)
            probs = F.softmax(logits, dim=1)
            predictions.append(probs)

        self.eval()  # Disable dropout

        # Stack predictions
        predictions = torch.stack(predictions)  # [mc_samples, batch, classes]

        # Variance across samples (epistemic uncertainty)
        epistemic_uncertainty = predictions.var(dim=0).mean(dim=1)

        return epistemic_uncertainty

    def predict_with_confidence(
        self,
        x: torch.Tensor,
        mc_samples: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make prediction with confidence and uncertainty estimates

        Args:
            x: Input tensor [batch, sequence, features]
            mc_samples: Number of MC samples for uncertainty

        Returns:
            Tuple of (predictions, probabilities, uncertainties)
        """
        self.eval()

        with torch.no_grad():
            logits, uncertainty, _ = self.forward(x, mc_samples=mc_samples)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        predictions_np = predictions.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()

        if uncertainty is not None:
            uncertainties_np = uncertainty.cpu().numpy()
        else:
            uncertainties_np = np.zeros(predictions_np.shape)

        return predictions_np, probabilities_np, uncertainties_np


class ModelEnsemble:
    """
    Ensemble of multiple model variants for robust predictions
    """

    def __init__(self, models: list):
        """
        Initialize ensemble

        Args:
            models: List of trained models
        """
        self.models = models

    def predict(self, x: torch.Tensor) -> Tuple[int, float, float]:
        """
        Ensemble prediction with voting

        Args:
            x: Input tensor

        Returns:
            Tuple of (prediction, probability, ensemble_agreement)
        """
        predictions = []
        probabilities = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(x)
                probs = F.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1)

                predictions.append(pred.item())
                probabilities.append(probs[0].cpu().numpy())

        # Majority vote
        final_prediction = max(set(predictions), key=predictions.count)

        # Average probability
        probabilities = np.array(probabilities)
        avg_probability = probabilities.mean(axis=0)
        final_probability = avg_probability[final_prediction]

        # Ensemble agreement (what % agreed)
        agreement = predictions.count(final_prediction) / len(predictions)

        return final_prediction, final_probability, agreement
