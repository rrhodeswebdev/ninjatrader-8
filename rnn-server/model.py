import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

class TradingRNN(nn.Module):
    """
    Simple LSTM-based RNN for predicting trade signals
    """
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=1):
        super(TradingRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)

        # Take the last output from the sequence
        last_output = lstm_out[:, -1, :]

        # Pass through fully connected layers
        out = self.relu(self.fc1(last_output))
        out = self.sigmoid(self.fc2(out))

        return out

class TradingModel:
    """
    Wrapper class for training and prediction
    """
    def __init__(self, sequence_length=20):
        self.model = TradingRNN(input_size=4, hidden_size=64, num_layers=2)
        self.scaler = StandardScaler()
        self.sequence_length = sequence_length
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def prepare_data(self, df):
        """
        Prepare data for training
        Creates sequences and labels based on future price movement
        """
        # Extract features (OHLC)
        features = df[['open', 'high', 'low', 'close']].values

        # Scale the features
        features_scaled = self.scaler.fit_transform(features)

        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - self.sequence_length):
            X.append(features_scaled[i:i + self.sequence_length])

            # Label: 1 if price goes up, 0 if it goes down
            # Look ahead 1 bar to see if we should have entered
            current_close = features[i + self.sequence_length - 1, 3]  # close price
            future_close = features[i + self.sequence_length, 3]  # next close
            y.append(1.0 if future_close > current_close else 0.0)

        return np.array(X), np.array(y)

    def train(self, df, epochs=50, learning_rate=0.001):
        """
        Train the model on historical data
        """
        print(f"\n{'='*50}")
        print("TRAINING RNN MODEL")
        print(f"{'='*50}")

        # Prepare data
        X, y = self.prepare_data(df)
        print(f"Training samples: {len(X)}")
        print(f"Sequence length: {self.sequence_length}")

        if len(X) < 10:
            print("WARNING: Not enough data to train effectively!")
            return

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        # Training setup
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        self.is_trained = True
        print(f"{'='*50}")
        print("MODEL TRAINING COMPLETE")
        print(f"{'='*50}\n")

    def predict(self, recent_bars_df):
        """
        Predict trade signal for new bar
        Returns: (signal, confidence)
            signal: 'long', 'short', or 'hold'
            confidence: float between 0 and 1
        """
        if not self.is_trained:
            print("WARNING: Model not trained yet!")
            return 'hold', 0.0

        # Need at least sequence_length bars for prediction
        if len(recent_bars_df) < self.sequence_length:
            print(f"WARNING: Need {self.sequence_length} bars for prediction, got {len(recent_bars_df)}")
            return 'hold', 0.0

        # Take last sequence_length bars
        recent_data = recent_bars_df[['open', 'high', 'low', 'close']].tail(self.sequence_length).values

        # Scale the data
        recent_scaled = self.scaler.transform(recent_data)

        # Convert to tensor
        X = torch.FloatTensor(recent_scaled).unsqueeze(0).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X)
            confidence = prediction.item()

        # Determine signal based on confidence
        # High confidence (>0.6) = long signal
        # Low confidence (<0.4) = short signal
        # Medium confidence = hold
        if confidence > 0.6:
            signal = 'long'
        elif confidence < 0.4:
            signal = 'short'
        else:
            signal = 'hold'

        return signal, confidence

# Global model instance
trading_model = TradingModel(sequence_length=20)
