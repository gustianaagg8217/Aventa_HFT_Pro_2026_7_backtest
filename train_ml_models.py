"""
Aventa HFT Pro 2026 - ML Model Training Script
Train LSTM/GRU/Transformer models with historical data for improved signal prediction
"""

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from sklearn.model_selection import train_test_split
from ml_neural_networks import LSTMPredictor, GRUPredictor, TransformerPredictor, EnsemblePredictor
from ml_predictor import FeatureEngineering
import logging
from datetime import datetime, timedelta
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(symbol: str, days: int = 90) -> pd.DataFrame:
    """
    Load historical OHLCV data for training
    
    Args:
        symbol: Trading symbol
        days: Number of days of history to load
        
    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Loading {days} days of historical data for {symbol}...")
    
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        return None
    
    try:
        # Request historical bars
        bars_needed = days * 24 * 60  # M1 bars per day
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, bars_needed)
        
        if rates is None or len(rates) == 0:
            logger.error(f"No data received for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        logger.info(f"✓ Loaded {len(df)} bars ({len(df) / (24*60):.1f} days)")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
    finally:
        mt5.shutdown()


def prepare_training_data(df: pd.DataFrame, config: dict = None) -> tuple:
    """
    Prepare features and labels for training
    
    Args:
        df: DataFrame with OHLCV data
        config: Configuration dictionary
        
    Returns:
        (X_train, X_val, y_train, y_val) for training
    """
    config = config or {}
    
    logger.info("Preparing features...")
    
    # Calculate technical features
    df = FeatureEngineering.calculate_technical_features(df)
    
    # Create target variable (future price direction)
    prediction_horizon = config.get('prediction_horizon', 5)  # bars ahead
    df['future_return'] = df['close'].shift(-prediction_horizon) / df['close'] - 1
    
    # Label: 1 for BUY (positive return), 0 for SELL (negative return)
    threshold = config.get('label_threshold', 0.0001)  # 0.01% minimum return
    df['label'] = 0
    df.loc[df['future_return'] > threshold, 'label'] = 1
    df.loc[df['future_return'] < -threshold, 'label'] = 0
    
    # Remove rows with NaN from indicator calculation
    df = df.dropna()
    
    # Balance classes (equal BUY and SELL signals)
    buy_samples = df[df['label'] == 1]
    sell_samples = df[df['label'] == 0]
    min_samples = min(len(buy_samples), len(sell_samples))
    
    df_balanced = pd.concat([
        buy_samples.sample(min_samples, random_state=42),
        sell_samples.sample(min_samples, random_state=42)
    ]).sort_index()
    
    logger.info(f"✓ Balanced dataset: {len(df_balanced)} samples")
    logger.info(f"  BUY signals: {(df_balanced['label'] == 1).sum()}")
    logger.info(f"  SELL signals: {(df_balanced['label'] == 0).sum()}")
    
    # Select features for model
    feature_columns = [col for col in df_balanced.columns if any(x in col for x in [
        'momentum', 'roc', 'sma', 'ema', 'volatility', 'atr', 'volume', 'spread',
        'returns', 'price_position', 'acceleration'
    ])]
    
    logger.info(f"✓ Using {len(feature_columns)} features")
    
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        df_balanced[feature_columns].values,
        df_balanced['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df_balanced['label']
    )
    
    logger.info(f"✓ Train set: {len(X_train)} samples")
    logger.info(f"✓ Validation set: {len(X_val)} samples")
    
    return X_train, X_val, y_train, y_val, feature_columns, df_balanced


def train_lstm_model(symbol: str, X_train, X_val, y_train, y_val, 
                     sequence_length: int = 60, save_path: str = "./models"):
    """Train LSTM model"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING LSTM MODEL")
    logger.info("="*60)
    
    # Initialize predictor
    lstm = LSTMPredictor(symbol, sequence_length=sequence_length)
    lstm.build_model(input_features=X_train.shape[1])
    
    # Prepare sequences
    df_train = pd.DataFrame(X_train)
    df_val = pd.DataFrame(X_val)
    df_train['label'] = y_train
    df_val['label'] = y_val
    
    feature_cols = [col for col in df_train.columns if col != 'label']
    X_train_seq, y_train_seq = lstm.prepare_sequences(df_train, feature_cols)
    X_val_seq, y_val_seq = lstm.prepare_sequences(df_val, feature_cols)
    
    logger.info(f"Training sequences: {X_train_seq.shape}")
    logger.info(f"Validation sequences: {X_val_seq.shape}")
    
    # Train
    history = lstm.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50)
    
    # Save
    os.makedirs(save_path, exist_ok=True)
    lstm.save(os.path.join(save_path, f'{symbol}_lstm.h5'))
    
    logger.info(f"✓ LSTM model saved to {save_path}")
    return lstm


def train_gru_model(symbol: str, X_train, X_val, y_train, y_val,
                    sequence_length: int = 60, save_path: str = "./models"):
    """Train GRU model"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING GRU MODEL")
    logger.info("="*60)
    
    gru = GRUPredictor(symbol, sequence_length=sequence_length)
    gru.build_model(input_features=X_train.shape[1])
    
    df_train = pd.DataFrame(X_train)
    df_val = pd.DataFrame(X_val)
    df_train['label'] = y_train
    df_val['label'] = y_val
    
    feature_cols = [col for col in df_train.columns if col != 'label']
    X_train_seq, y_train_seq = gru.prepare_sequences(df_train, feature_cols)
    X_val_seq, y_val_seq = gru.prepare_sequences(df_val, feature_cols)
    
    logger.info(f"Training sequences: {X_train_seq.shape}")
    logger.info(f"Validation sequences: {X_val_seq.shape}")
    
    history = gru.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50)
    
    os.makedirs(save_path, exist_ok=True)
    gru.save(os.path.join(save_path, f'{symbol}_gru.h5'))
    
    logger.info(f"✓ GRU model saved to {save_path}")
    return gru


def train_transformer_model(symbol: str, X_train, X_val, y_train, y_val,
                           sequence_length: int = 60, save_path: str = "./models"):
    """Train Transformer model"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING TRANSFORMER MODEL")
    logger.info("="*60)
    
    transformer = TransformerPredictor(symbol, sequence_length=sequence_length)
    transformer.build_model(input_features=X_train.shape[1])
    
    df_train = pd.DataFrame(X_train)
    df_val = pd.DataFrame(X_val)
    df_train['label'] = y_train
    df_val['label'] = y_val
    
    feature_cols = [col for col in df_train.columns if col != 'label']
    X_train_seq, y_train_seq = transformer.prepare_sequences(df_train, feature_cols)
    X_val_seq, y_val_seq = transformer.prepare_sequences(df_val, feature_cols)
    
    logger.info(f"Training sequences: {X_train_seq.shape}")
    logger.info(f"Validation sequences: {X_val_seq.shape}")
    
    history = transformer.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50)
    
    os.makedirs(save_path, exist_ok=True)
    transformer.save(os.path.join(save_path, f'{symbol}_transformer.h5'))
    
    logger.info(f"✓ Transformer model saved to {save_path}")
    return transformer


def train_ensemble(symbol: str, X_train, X_val, y_train, y_val,
                  sequence_length: int = 60, save_path: str = "./models"):
    """Train ensemble model"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING ENSEMBLE (LSTM + GRU + TRANSFORMER)")
    logger.info("="*60)
    
    ensemble = EnsemblePredictor(symbol, sequence_length=sequence_length)
    ensemble.build_models(input_features=X_train.shape[1])
    
    df_train = pd.DataFrame(X_train)
    df_val = pd.DataFrame(X_val)
    df_train['label'] = y_train
    df_val['label'] = y_val
    
    feature_cols = [col for col in df_train.columns if col != 'label']
    X_train_seq, y_train_seq = ensemble.lstm.prepare_sequences(df_train, feature_cols)
    X_val_seq, y_val_seq = ensemble.lstm.prepare_sequences(df_val, feature_cols)
    
    logger.info(f"Training sequences: {X_train_seq.shape}")
    
    # Train all models
    ensemble.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50)
    
    os.makedirs(save_path, exist_ok=True)
    ensemble.save(os.path.join(save_path, f'{symbol}_ensemble'))
    
    logger.info(f"✓ Ensemble models saved to {save_path}")
    return ensemble


def main():
    """Main training script"""
    
    # Configuration
    SYMBOL = 'EURUSD'  # Change to your symbol
    DAYS_OF_HISTORY = 90
    SEQUENCE_LENGTH = 60
    MODELS_PATH = './models'
    
    logger.info("="*60)
    logger.info("AVENTA HFT PRO 2026 - ML MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"History: {DAYS_OF_HISTORY} days")
    logger.info(f"Sequence length: {SEQUENCE_LENGTH} minutes")
    logger.info(f"Models path: {MODELS_PATH}")
    
    # Load data
    df = load_training_data(SYMBOL, DAYS_OF_HISTORY)
    if df is None:
        logger.error("Failed to load data")
        return
    
    # Prepare features
    X_train, X_val, y_train, y_val, feature_cols, df_balanced = prepare_training_data(df)
    
    # Train individual models
    logger.info("\nTraining individual models...")
    lstm = train_lstm_model(SYMBOL, X_train, X_val, y_train, y_val, SEQUENCE_LENGTH, MODELS_PATH)
    gru = train_gru_model(SYMBOL, X_train, X_val, y_train, y_val, SEQUENCE_LENGTH, MODELS_PATH)
    transformer = train_transformer_model(SYMBOL, X_train, X_val, y_train, y_val, SEQUENCE_LENGTH, MODELS_PATH)
    
    # Train ensemble
    logger.info("\nTraining ensemble...")
    ensemble = train_ensemble(SYMBOL, X_train, X_val, y_train, y_val, SEQUENCE_LENGTH, MODELS_PATH)
    
    logger.info("\n" + "="*60)
    logger.info("✓ ALL MODELS TRAINED SUCCESSFULLY")
    logger.info("="*60)
    logger.info(f"Models saved in: {MODELS_PATH}/")
    logger.info("\nNext steps:")
    logger.info("1. Load ensemble in aventa_hft_core.py")
    logger.info("2. Use ensemble.predict() for trading signals")
    logger.info("3. Monitor model performance in live trading")
    logger.info("4. Retrain monthly with new data")


if __name__ == "__main__":
    main()
