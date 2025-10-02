import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
from pathlib import Path
import json
from hurst import compute_Hc
from scipy import stats

def calculate_hurst_exponent(prices, min_window=10):
    """
    Calculate Hurst exponent for time series using the Mottl/hurst library
    H < 0.5: Mean reverting (anti-persistent)
    H = 0.5: Random walk (Brownian motion)
    H > 0.5: Trending (persistent)

    Returns both H and C (Hurst exponent and constant from fit)
    """
    if len(prices) < min_window:
        return 0.5, 1.0  # Default to random walk if insufficient data

    try:
        # compute_Hc returns (H, c, data) where:
        # H = Hurst exponent
        # c = Constant from the fit
        # data = (x, y) values used for fitting
        H, c, _ = compute_Hc(prices, kind='price', simplified=True)

        # Clamp H between 0 and 1 for safety
        H = max(0.0, min(1.0, H))

        return H, c
    except Exception as e:
        # Fallback to default if computation fails
        print(f"Warning: Hurst calculation failed: {e}")
        return 0.5, 1.0


def calculate_atr(high, low, close, period=14):
    """
    Calculate Average True Range
    Measures market volatility
    """
    if len(high) < period + 1:
        return np.zeros(len(high))

    # True Range calculation
    tr_list = []
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr = max(hl, hc, lc)
        tr_list.append(tr)

    # ATR is moving average of TR
    atr = []
    for i in range(len(tr_list)):
        if i < period - 1:
            atr.append(np.mean(tr_list[:i + 1]))
        else:
            atr.append(np.mean(tr_list[i - period + 1:i + 1]))

    # Add zero for first value (no previous close)
    atr = [0.0] + atr

    return np.array(atr)


def calculate_price_features(df):
    """
    Calculate advanced price-based features (no typical indicators)
    Returns dictionary of feature arrays
    """
    ohlc = df[['open', 'high', 'low', 'close']].values
    n_bars = len(ohlc)

    # Initialize feature dictionary
    features = {}

    # 1. Price Momentum (Velocity & Acceleration)
    velocity = np.zeros(n_bars)
    acceleration = np.zeros(n_bars)
    lookback = 5
    for i in range(lookback, n_bars):
        velocity[i] = (ohlc[i, 3] - ohlc[i-lookback, 3]) / lookback
        if i > lookback:
            acceleration[i] = velocity[i] - velocity[i-1]
    features['velocity'] = velocity
    features['acceleration'] = acceleration

    # 2. Range Dynamics (Range Ratio & Wick Ratio)
    range_ratio = np.ones(n_bars)
    wick_ratio = np.zeros(n_bars)
    for i in range(1, n_bars):
        current_range = ohlc[i, 1] - ohlc[i, 2]  # high - low
        prev_range = ohlc[i-1, 1] - ohlc[i-1, 2]
        range_ratio[i] = current_range / (prev_range + 1e-8)

        # Wick calculations
        upper_wick = ohlc[i, 1] - max(ohlc[i, 0], ohlc[i, 3])  # high - max(open, close)
        lower_wick = min(ohlc[i, 0], ohlc[i, 3]) - ohlc[i, 2]  # min(open, close) - low
        wick_ratio[i] = upper_wick / (lower_wick + 1e-8)
    features['range_ratio'] = range_ratio
    features['wick_ratio'] = wick_ratio

    # 3. Gap Analysis
    gap_up = np.zeros(n_bars)
    gap_down = np.zeros(n_bars)
    gap_filled = np.zeros(n_bars)
    for i in range(1, n_bars):
        gap_up[i] = max(0, ohlc[i, 2] - ohlc[i-1, 1])  # low - prev high
        gap_down[i] = max(0, ohlc[i-1, 2] - ohlc[i, 1])  # prev low - high
        # Check if gap filled (prev close within current range)
        if ohlc[i, 2] <= ohlc[i-1, 3] <= ohlc[i, 1]:
            gap_filled[i] = 1
    features['gap_up'] = gap_up
    features['gap_down'] = gap_down
    features['gap_filled'] = gap_filled

    # 4. Price Fractals (Swing Highs/Lows)
    swing_high = np.zeros(n_bars)
    swing_low = np.zeros(n_bars)
    bars_since_swing_high = np.zeros(n_bars)
    bars_since_swing_low = np.zeros(n_bars)

    last_swing_high_idx = 0
    last_swing_low_idx = 0

    for i in range(2, n_bars - 1):
        # Swing high: higher than neighbors
        if ohlc[i, 1] > ohlc[i-1, 1] and ohlc[i, 1] > ohlc[i+1, 1]:
            swing_high[i] = 1
            last_swing_high_idx = i
        # Swing low: lower than neighbors
        if ohlc[i, 2] < ohlc[i-1, 2] and ohlc[i, 2] < ohlc[i+1, 2]:
            swing_low[i] = 1
            last_swing_low_idx = i

        bars_since_swing_high[i] = i - last_swing_high_idx
        bars_since_swing_low[i] = i - last_swing_low_idx

    features['swing_high'] = swing_high
    features['swing_low'] = swing_low
    features['bars_since_swing_high'] = bars_since_swing_high
    features['bars_since_swing_low'] = bars_since_swing_low

    # 5. Return Distribution (Skewness & Kurtosis)
    skewness = np.zeros(n_bars)
    kurtosis_vals = np.zeros(n_bars)
    window = 20

    returns = np.diff(ohlc[:, 3]) / ohlc[:-1, 3]
    for i in range(window, n_bars):
        recent_returns = returns[i-window:i]
        if len(recent_returns) > 3:  # Need minimum for stats
            skewness[i] = stats.skew(recent_returns)
            kurtosis_vals[i] = stats.kurtosis(recent_returns)
    features['skewness'] = skewness
    features['kurtosis'] = kurtosis_vals

    # 6. Rolling Min/Max Distance (Position in Range)
    position_in_range = np.zeros(n_bars)
    window = 20
    for i in range(window, n_bars):
        rolling_max = np.max(ohlc[i-window:i+1, 3])
        rolling_min = np.min(ohlc[i-window:i+1, 3])
        range_size = rolling_max - rolling_min
        if range_size > 1e-8:
            position_in_range[i] = (ohlc[i, 3] - rolling_min) / range_size
    features['position_in_range'] = position_in_range

    # 7. Trend Structure (Higher Highs / Lower Lows count)
    higher_highs = np.zeros(n_bars)
    lower_lows = np.zeros(n_bars)
    trend_strength = np.zeros(n_bars)
    window = 5

    for i in range(window, n_bars):
        hh_count = sum([1 for j in range(i-window+1, i+1) if ohlc[j, 1] > ohlc[j-1, 1]])
        ll_count = sum([1 for j in range(i-window+1, i+1) if ohlc[j, 2] < ohlc[j-1, 2]])
        higher_highs[i] = hh_count
        lower_lows[i] = ll_count
        trend_strength[i] = hh_count - ll_count  # -5 to +5

    features['higher_highs'] = higher_highs
    features['lower_lows'] = lower_lows
    features['trend_strength'] = trend_strength

    return features


class TradingRNN(nn.Module):
    """
    LSTM-based RNN for predicting trade signals with 3-class output
    Features (24 total):
    - OHLC (4)
    - Hurst H & C (2)
    - ATR (1)
    - Velocity & Acceleration (2)
    - Range Ratio & Wick Ratio (2)
    - Gap Up/Down/Filled (3)
    - Swing High/Low & Bars Since (4)
    - Skewness & Kurtosis (2)
    - Position in Range (1)
    - Higher Highs/Lower Lows/Trend Strength (3)
    """
    def __init__(self, input_size=24, hidden_size=64, num_layers=2, output_size=3):
        super(TradingRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers with dropout between layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)

        # Take the last output from the sequence
        last_output = lstm_out[:, -1, :]

        # Pass through fully connected layers with dropout
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)  # No activation - softmax applied in loss function

        return out

class TradingModel:
    """
    Wrapper class for training and prediction with state management
    """
    def __init__(self, sequence_length=20, model_path='models/trading_model.pth'):
        self.model = TradingRNN(input_size=24, hidden_size=64, num_layers=2, output_size=3)
        self.scaler = StandardScaler()
        self.sequence_length = sequence_length
        self.is_trained = False
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {self.device}")
        self.model.to(self.device)

        # Compile model for faster inference (PyTorch 2.0+)
        try:
            self.model = torch.compile(self.model)
            print("Model compiled for optimized inference")
        except Exception as e:
            print(f"Model compilation not available: {e}")

        # Try to load existing model
        if self.model_path.exists():
            self.load_model()

        # Historical data storage
        self.historical_data = None

    def _validate_data(self, df):
        """Validate input data for NaN, inf, and required columns"""
        required_cols = ['open', 'high', 'low', 'close']

        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        if df[required_cols].isnull().any().any():
            raise ValueError("Input data contains NaN values")

        if np.isinf(df[required_cols].values).any():
            raise ValueError("Input data contains infinite values")

        if len(df) < self.sequence_length + 1:
            raise ValueError(f"Need at least {self.sequence_length + 1} bars, got {len(df)}")

    def prepare_data(self, df, fit_scaler=False, adaptive_threshold=True):
        """
        Prepare data for training with enhanced features
        Creates sequences and labels based on future price movement

        Args:
            df: DataFrame with OHLC data
            fit_scaler: If True, fit the scaler. If False, only transform (use False for inference)
            adaptive_threshold: If True, calculate threshold based on data volatility
        """
        self._validate_data(df)

        # Extract OHLC features
        ohlc = df[['open', 'high', 'low', 'close']].values

        # Calculate adaptive threshold based on price volatility
        if adaptive_threshold or not hasattr(self, 'signal_threshold'):
            price_changes = np.diff(ohlc[:, 3]) / ohlc[:-1, 3] * 100  # % changes
            volatility = np.std(price_changes)

            # Threshold = 0.5x volatility (adjust multiplier as needed)
            # This ensures ~60-70% of data is not HOLD
            self.signal_threshold = max(0.01, volatility * 0.5)

            print(f"Data volatility: {volatility:.4f}%")
            print(f"Adaptive threshold set to: {self.signal_threshold:.4f}%")

        # Calculate Hurst exponent (H) and constant (c) over rolling window
        # Note: hurst library requires minimum 100 data points
        hurst_H_values = []
        hurst_C_values = []
        for i in range(len(df)):
            if i < 100:  # Need minimum 100 bars for hurst library
                hurst_H_values.append(0.5)
                hurst_C_values.append(1.0)
            else:
                # Use last 100 bars for Hurst calculation
                prices = df['close'].iloc[i-99:i+1].values
                H, c = calculate_hurst_exponent(prices)
                hurst_H_values.append(H)
                hurst_C_values.append(c)

        # Log Hurst statistics for the dataset
        hurst_valid = [h for h in hurst_H_values if h != 0.5]
        if len(hurst_valid) > 0:
            avg_hurst = np.mean(hurst_valid)
            print(f"Hurst exponent statistics:")
            print(f"  Mean H: {avg_hurst:.4f}")
            print(f"  Min H: {min(hurst_valid):.4f}")
            print(f"  Max H: {max(hurst_valid):.4f}")
            if avg_hurst > 0.5:
                print(f"  → Overall TRENDING (persistent) market")
            elif avg_hurst < 0.5:
                print(f"  → Overall MEAN-REVERTING (anti-persistent) market")
            else:
                print(f"  → Overall RANDOM WALK market")

        # Calculate ATR
        atr = calculate_atr(df['high'].values, df['low'].values, df['close'].values)

        # Calculate all advanced price features
        price_features = calculate_price_features(df)

        # Combine all features (25 total):
        # OHLC (4) + Hurst (2) + ATR (1) + Price Features (18) = 25
        features = np.column_stack([
            ohlc,                                    # 4
            hurst_H_values,                          # 1
            hurst_C_values,                          # 1
            atr,                                     # 1
            price_features['velocity'],              # 1
            price_features['acceleration'],          # 1
            price_features['range_ratio'],           # 1
            price_features['wick_ratio'],            # 1
            price_features['gap_up'],                # 1
            price_features['gap_down'],              # 1
            price_features['gap_filled'],            # 1
            price_features['swing_high'],            # 1
            price_features['swing_low'],             # 1
            price_features['bars_since_swing_high'], # 1
            price_features['bars_since_swing_low'],  # 1
            price_features['skewness'],              # 1
            price_features['kurtosis'],              # 1
            price_features['position_in_range'],     # 1
            price_features['higher_highs'],          # 1
            price_features['lower_lows'],            # 1
            price_features['trend_strength']         # 1
        ])

        print(f"Total features: {features.shape[1]} (OHLC:4 + Hurst:2 + ATR:1 + Price:18)")

        # Scale the features
        if fit_scaler:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)

        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - self.sequence_length):
            X.append(features_scaled[i:i + self.sequence_length])

            # 3-class label: 0=short, 1=hold, 2=long
            # Look ahead 1 bar to see price movement
            current_close = ohlc[i + self.sequence_length - 1, 3]
            future_close = ohlc[i + self.sequence_length, 3]

            price_change_pct = (future_close - current_close) / current_close * 100

            # Use adaptive threshold
            if price_change_pct > self.signal_threshold:
                y.append(2)  # Long
            elif price_change_pct < -self.signal_threshold:
                y.append(0)  # Short
            else:
                y.append(1)  # Hold

        return np.array(X), np.array(y)

    def train(self, df, epochs=100, learning_rate=0.001, batch_size=32, validation_split=0.2):
        """
        Train the model on historical data with validation split and early stopping
        """
        print(f"\n{'='*50}")
        print("TRAINING RNN MODEL")
        print(f"{'='*50}")

        # Prepare data with scaler fitting
        X, y = self.prepare_data(df, fit_scaler=True)
        print(f"Total samples: {len(X)}")
        print(f"Sequence length: {self.sequence_length}")

        if len(X) < 20:
            print("WARNING: Not enough data to train effectively!")
            return

        # Check if we can stratify (need at least 2 samples per class)
        unique, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()
        can_stratify = min_samples >= 2 and len(X) >= 10

        print(f"Class distribution: {dict(zip(unique, counts))}")

        # Train/validation split
        if can_stratify:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            print("Using stratified split")
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
            print("Using non-stratified split (insufficient samples per class)")

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

        # Create DataLoader for mini-batch training
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Validation tensors
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)

        # Calculate class weights to handle imbalance
        class_counts = np.bincount(y_train, minlength=3)
        total_samples = len(y_train)
        class_weights = torch.FloatTensor([
            total_samples / (3 * max(count, 1)) for count in class_counts
        ]).to(self.device)

        print(f"Class weights: Short={class_weights[0]:.2f}, Hold={class_weights[1]:.2f}, Long={class_weights[2]:.2f}")

        # Training setup with weighted loss
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Early stopping setup
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_train_loss = epoch_loss / batch_count

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)

                # Calculate metrics
                val_predictions = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_accuracy = accuracy_score(y_val, val_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_val, val_predictions, average='weighted', zero_division=0
                )

            self.model.train()

            # Learning rate scheduling
            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                # Show prediction distribution
                pred_dist = np.bincount(val_predictions, minlength=3)
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
                print(f"  Val Accuracy: {val_accuracy:.4f}, F1: {f1:.4f}")
                print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
                print(f"  Predictions: Short={pred_dist[0]}, Hold={pred_dist[1]}, Long={pred_dist[2]}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val_tensor)
            val_predictions = torch.argmax(val_outputs, dim=1).cpu().numpy()

            print(f"\n{'='*50}")
            print("FINAL VALIDATION METRICS")
            print(f"{'='*50}")
            print(f"Accuracy: {accuracy_score(y_val, val_predictions):.4f}")

            # Create confusion matrix with all labels to ensure correct shape
            cm = confusion_matrix(y_val, val_predictions, labels=[0, 1, 2])

            print("\nConfusion Matrix:")
            print("              Predicted")
            print("              Short  Hold  Long")
            print(f"Actual Short   {cm[0][0]:5}  {cm[0][1]:5}  {cm[0][2]:5}")
            print(f"       Hold    {cm[1][0]:5}  {cm[1][1]:5}  {cm[1][2]:5}")
            print(f"       Long    {cm[2][0]:5}  {cm[2][1]:5}  {cm[2][2]:5}")

            # Print class distribution
            unique_val, counts_val = np.unique(y_val, return_counts=True)
            unique_pred, counts_pred = np.unique(val_predictions, return_counts=True)
            print(f"\nValidation set distribution: {dict(zip(unique_val, counts_val))}")
            print(f"Predictions distribution: {dict(zip(unique_pred, counts_pred))}")

        self.is_trained = True
        print(f"\n{'='*50}")
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

        # Calculate current Hurst exponent for logging
        current_hurst_H = 0.5
        current_hurst_C = 1.0
        if len(recent_bars_df) >= 100:
            try:
                prices = recent_bars_df['close'].tail(100).values
                current_hurst_H, current_hurst_C = calculate_hurst_exponent(prices)
            except Exception as e:
                print(f"Warning: Could not calculate current Hurst: {e}")

        # Validate and prepare data (without fitting scaler)
        try:
            # Need full historical data for Hurst calculation
            X, _ = self.prepare_data(recent_bars_df, fit_scaler=False)

            if len(X) == 0:
                print(f"WARNING: Need at least {self.sequence_length + 1} bars for prediction")
                return 'hold', 0.0

            # Take the last sequence for prediction
            last_sequence = X[-1:]

        except Exception as e:
            print(f"ERROR preparing prediction data: {e}")
            return 'hold', 0.0

        # Convert to tensor
        X_tensor = torch.FloatTensor(last_sequence).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

            # Get predicted class and confidence
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        # Map class to signal: 0=short, 1=hold, 2=long
        signal_map = {0: 'short', 1: 'hold', 2: 'long'}
        signal = signal_map[predicted_class]

        # Log Hurst values with prediction
        print(f"\n--- Prediction Context ---")
        print(f"Current Hurst H: {current_hurst_H:.4f} ", end="")
        if current_hurst_H > 0.5:
            print("(TRENDING)")
        elif current_hurst_H < 0.5:
            print("(MEAN-REVERTING)")
        else:
            print("(RANDOM WALK)")
        print(f"Current Hurst C: {current_hurst_C:.4f}")
        print(f"Predicted Signal: {signal.upper()}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"-------------------------\n")

        return signal, confidence

    def save_model(self, path=None):
        """Save model state, scaler, and configuration"""
        if path is None:
            path = self.model_path
        else:
            path = Path(path)

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
            'scaler_var': self.scaler.var_ if hasattr(self.scaler, 'var_') else None,
            'sequence_length': self.sequence_length,
            'is_trained': self.is_trained,
            'signal_threshold': self.signal_threshold if hasattr(self, 'signal_threshold') else 0.05,
        }

        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path=None):
        """Load model state, scaler, and configuration"""
        if path is None:
            path = self.model_path
        else:
            path = Path(path)

        if not path.exists():
            print(f"No model found at {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load scaler state
            if checkpoint['scaler_mean'] is not None:
                self.scaler.mean_ = checkpoint['scaler_mean']
                self.scaler.scale_ = checkpoint['scaler_scale']
                self.scaler.var_ = checkpoint['scaler_var']
                self.scaler.n_features_in_ = len(checkpoint['scaler_mean'])
                self.scaler.n_samples_seen_ = 1  # Dummy value

            # Load configuration
            self.sequence_length = checkpoint['sequence_length']
            self.is_trained = checkpoint['is_trained']
            self.signal_threshold = checkpoint.get('signal_threshold', 0.05)

            print(f"Model loaded from {path}")
            print(f"Signal threshold: {self.signal_threshold:.4f}%")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def update_historical_data(self, df):
        """Update historical data storage"""
        if self.historical_data is None:
            self.historical_data = df.copy()
        else:
            self.historical_data = pd.concat([self.historical_data, df], ignore_index=True)

        # Keep only recent data to prevent memory issues (e.g., last 50,000 bars)
        # At 1-min bars: 50k bars = ~35 days, 5-min bars = ~6 months, daily bars = ~137 years
        if len(self.historical_data) > 50000:
            self.historical_data = self.historical_data.tail(50000).reset_index(drop=True)

        return self.historical_data

# Global model instance
trading_model = TradingModel(sequence_length=20)
