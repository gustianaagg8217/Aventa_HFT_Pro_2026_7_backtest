"""
Aventa HFT Pro 2026 - Neural Network Models (LSTM/GRU)
Advanced deep learning for improved price prediction (65-70%+ accuracy target)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


class LSTMPredictor:
    """LSTM-based price direction predictor"""
    
    def __init__(self, symbol: str, sequence_length: int = 60, config: Optional[Dict] = None):
        """
        Initialize LSTM predictor
        
        Args:
            symbol: Trading symbol
            sequence_length: Number of past bars to use for prediction (default 60 = 1 hour M1)
            config: Configuration dictionary
        """
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.config = config or {}
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        
    def build_model(self, input_features: int = 20, lstm_units: int = 128) -> keras.Model:
        """
        Build LSTM neural network
        
        Architecture:
        - Input: (batch, sequence_length, input_features)
        - LSTM 1: 128 units with dropout
        - LSTM 2: 64 units with dropout
        - Dense: 32 units
        - Output: Binary classification (BUY=1, SELL=0)
        """
        model = Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, input_features)),
            
            # First LSTM layer
            layers.LSTM(lstm_units, activation='relu', return_sequences=True, 
                       kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            
            # Second LSTM layer
            layers.LSTM(lstm_units // 2, activation='relu', return_sequences=False,
                       kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            
            # Output layer for binary classification
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        self.model = model
        logger.info(f"LSTM model built with {input_features} input features")
        return model
    
    def prepare_sequences(self, data: pd.DataFrame, features: List[str], 
                         label_column: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training
        
        Args:
            data: DataFrame with features and labels
            features: List of feature columns to use
            label_column: Name of label column
            
        Returns:
            X: Array of shape (samples, sequence_length, num_features)
            y: Array of shape (samples,) with binary labels
        """
        # Extract feature data
        feature_data = data[features].values
        labels = data[label_column].values
        
        # Normalize features
        feature_data = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(feature_data) - self.sequence_length):
            sequence = feature_data[i:i + self.sequence_length]
            label = labels[i + self.sequence_length]
            X.append(sequence)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Train LSTM model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Dictionary with training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        
        logger.info(f"Training LSTM on {len(X_train)} samples...")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self.is_trained = True
        logger.info("LSTM training completed")
        
        return history.history
    
    def predict(self, sequence: np.ndarray) -> Tuple[float, float]:
        """
        Predict signal for a given sequence
        
        Args:
            sequence: Array of shape (sequence_length, num_features)
            
        Returns:
            (signal, confidence) where signal is 0 or 1 and confidence is 0-1
        """
        if self.model is None or not self.is_trained:
            raise ValueError("Model not trained")
        
        # Normalize
        sequence_normalized = self.scaler.transform(sequence)
        
        # Add batch dimension
        sequence_batch = np.expand_dims(sequence_normalized, axis=0)
        
        # Predict
        prediction = self.model.predict(sequence_batch, verbose=0)[0][0]
        
        # Convert to signal: 1 for BUY (confidence > 0.5), 0 for SELL
        signal = 1 if prediction > 0.5 else 0
        confidence = max(prediction, 1 - prediction)
        
        return signal, confidence
    
    def save(self, filepath: str):
        """Save model and scaler"""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and scaler"""
        self.model = keras.models.load_model(filepath)
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class GRUPredictor:
    """GRU-based price direction predictor (lighter than LSTM)"""
    
    def __init__(self, symbol: str, sequence_length: int = 60, config: Optional[Dict] = None):
        """
        Initialize GRU predictor
        
        Args:
            symbol: Trading symbol
            sequence_length: Number of past bars
            config: Configuration dictionary
        """
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.config = config or {}
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
    
    def build_model(self, input_features: int = 20, gru_units: int = 64) -> keras.Model:
        """
        Build GRU neural network (computationally lighter than LSTM)
        
        Architecture:
        - Input: (batch, sequence_length, input_features)
        - GRU 1: 64 units with dropout
        - GRU 2: 32 units with dropout
        - Dense: 16 units
        - Output: Binary classification
        """
        model = Sequential([
            layers.Input(shape=(self.sequence_length, input_features)),
            
            layers.GRU(gru_units, activation='relu', return_sequences=True,
                      kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            
            layers.GRU(gru_units // 2, activation='relu', return_sequences=False,
                      kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        self.model = model
        logger.info(f"GRU model built with {input_features} input features")
        return model
    
    def prepare_sequences(self, data: pd.DataFrame, features: List[str],
                         label_column: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training"""
        feature_data = data[features].values
        labels = data[label_column].values
        
        feature_data = self.scaler.fit_transform(feature_data)
        
        X, y = [], []
        for i in range(len(feature_data) - self.sequence_length):
            sequence = feature_data[i:i + self.sequence_length]
            label = labels[i + self.sequence_length]
            X.append(sequence)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train GRU model"""
        if self.model is None:
            raise ValueError("Model not built")
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        
        logger.info(f"Training GRU on {len(X_train)} samples...")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self.is_trained = True
        return history.history
    
    def predict(self, sequence: np.ndarray) -> Tuple[float, float]:
        """Predict signal"""
        if self.model is None or not self.is_trained:
            raise ValueError("Model not trained")
        
        sequence_normalized = self.scaler.transform(sequence)
        sequence_batch = np.expand_dims(sequence_normalized, axis=0)
        prediction = self.model.predict(sequence_batch, verbose=0)[0][0]
        
        signal = 1 if prediction > 0.5 else 0
        confidence = max(prediction, 1 - prediction)
        
        return signal, confidence
    
    def save(self, filepath: str):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"GRU model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        self.model = keras.models.load_model(filepath)
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True


class TransformerPredictor:
    """Transformer-based predictor for advanced feature extraction"""
    
    def __init__(self, symbol: str, sequence_length: int = 60, num_heads: int = 4, config: Optional[Dict] = None):
        """
        Initialize Transformer predictor
        
        Args:
            symbol: Trading symbol
            sequence_length: Sequence length
            num_heads: Number of attention heads
            config: Configuration
        """
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.config = config or {}
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
    
    def build_model(self, input_features: int = 20, d_model: int = 64) -> keras.Model:
        """
        Build Transformer model
        
        Uses multi-head self-attention for better feature extraction
        """
        inputs = keras.Input(shape=(self.sequence_length, input_features))
        
        # Attention block 1
        attn_output = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=d_model // self.num_heads,
            kernel_regularizer=keras.regularizers.l2(0.001)
        )(inputs, inputs)
        attn_output = layers.Dropout(0.1)(attn_output)
        out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
        
        # Feed forward
        ffn_output = layers.Dense(256, activation="relu")(out1)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        ffn_output = layers.Dense(input_features)(ffn_output)
        out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
        # Global average pooling
        features = layers.GlobalAveragePooling1D()(out2)
        
        # Classification head
        x = layers.Dense(128, activation="relu")(features)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        self.model = model
        logger.info(f"Transformer model built with {input_features} input features")
        return model
    
    def prepare_sequences(self, data: pd.DataFrame, features: List[str],
                         label_column: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences"""
        feature_data = data[features].values
        labels = data[label_column].values
        
        feature_data = self.scaler.fit_transform(feature_data)
        
        X, y = [], []
        for i in range(len(feature_data) - self.sequence_length):
            sequence = feature_data[i:i + self.sequence_length]
            label = labels[i + self.sequence_length]
            X.append(sequence)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train Transformer"""
        if self.model is None:
            raise ValueError("Model not built")
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        
        logger.info(f"Training Transformer on {len(X_train)} samples...")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self.is_trained = True
        return history.history
    
    def predict(self, sequence: np.ndarray) -> Tuple[float, float]:
        """Predict signal"""
        if self.model is None or not self.is_trained:
            raise ValueError("Model not trained")
        
        sequence_normalized = self.scaler.transform(sequence)
        sequence_batch = np.expand_dims(sequence_normalized, axis=0)
        prediction = self.model.predict(sequence_batch, verbose=0)[0][0]
        
        signal = 1 if prediction > 0.5 else 0
        confidence = max(prediction, 1 - prediction)
        
        return signal, confidence
    
    def save(self, filepath: str):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Transformer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        self.model = keras.models.load_model(filepath)
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True


class EnsemblePredictor:
    """Ensemble combining LSTM, GRU, and Transformer for maximum accuracy"""
    
    def __init__(self, symbol: str, sequence_length: int = 60, config: Optional[Dict] = None):
        """Initialize ensemble"""
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.config = config or {}
        
        self.lstm = LSTMPredictor(symbol, sequence_length, config)
        self.gru = GRUPredictor(symbol, sequence_length, config)
        self.transformer = TransformerPredictor(symbol, sequence_length, config=config)
        
        # Ensemble weights (can be learned from validation set)
        self.lstm_weight = 0.4
        self.gru_weight = 0.3
        self.transformer_weight = 0.3
        
        self.is_trained = False
    
    def build_models(self, input_features: int = 20):
        """Build all component models"""
        logger.info("Building ensemble models...")
        self.lstm.build_model(input_features)
        self.gru.build_model(input_features)
        self.transformer.build_model(input_features)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, epochs: int = 50):
        """Train all models"""
        logger.info("Training ensemble...")
        
        self.lstm.train(X_train, y_train, X_val, y_val, epochs)
        self.gru.train(X_train, y_train, X_val, y_val, epochs)
        self.transformer.train(X_train, y_train, X_val, y_val, epochs)
        
        self.is_trained = True
        logger.info("Ensemble training completed")
    
    def predict(self, sequence: np.ndarray) -> Tuple[float, float]:
        """
        Predict using ensemble voting
        
        Returns:
            (signal, confidence) combined from all models
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained")
        
        lstm_signal, lstm_conf = self.lstm.predict(sequence)
        gru_signal, gru_conf = self.gru.predict(sequence)
        transformer_signal, transformer_conf = self.transformer.predict(sequence)
        
        # Weighted ensemble prediction
        ensemble_pred = (
            lstm_signal * self.lstm_weight * lstm_conf +
            gru_signal * self.gru_weight * gru_conf +
            transformer_signal * self.transformer_weight * transformer_conf
        ) / (self.lstm_weight + self.gru_weight + self.transformer_weight)
        
        signal = 1 if ensemble_pred > 0.5 else 0
        confidence = max(ensemble_pred, 1 - ensemble_pred)
        
        return signal, confidence
    
    def save(self, dirpath: str):
        """Save all models"""
        os.makedirs(dirpath, exist_ok=True)
        self.lstm.save(os.path.join(dirpath, 'lstm.h5'))
        self.gru.save(os.path.join(dirpath, 'gru.h5'))
        self.transformer.save(os.path.join(dirpath, 'transformer.h5'))
        logger.info(f"Ensemble saved to {dirpath}")
    
    def load(self, dirpath: str):
        """Load all models"""
        self.lstm.load(os.path.join(dirpath, 'lstm.h5'))
        self.gru.load(os.path.join(dirpath, 'gru.h5'))
        self.transformer.load(os.path.join(dirpath, 'transformer.h5'))
        self.is_trained = True
        logger.info(f"Ensemble loaded from {dirpath}")
