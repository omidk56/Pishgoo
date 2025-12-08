"""
AI/ML Engine - Machine Learning and Deep Learning Models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Optional imports - only import if available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle
import joblib
from utils.logger import setup_logger
from utils.indicators import TechnicalIndicators

logger = setup_logger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class AIModel:
    """AI/ML model for trading predictions"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize AI model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.indicators = TechnicalIndicators()
        self.scaler = StandardScaler()
        self.rf_model: Optional[RandomForestClassifier] = None
        self.xgb_model = None  # Optional[xgb.XGBClassifier] if XGBOOST_AVAILABLE
        self.lstm_model = None  # Optional[tf.keras.Model] if TENSORFLOW_AVAILABLE
        self.feature_columns: List[str] = []
        self.sequence_length = 60  # For LSTM
        logger.info(" AI Model initialized")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and targets from DataFrame
        
        Args:
            df: DataFrame with OHLCV and indicators
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if df.empty or len(df) < 50:
            logger.warning(" Insufficient data for feature preparation")
            return pd.DataFrame(), pd.Series(dtype=float)
        
        # Calculate indicators if not present
        if 'rsi' not in df.columns:
            df = self.indicators.calculate_all(df)
        
        # Create features
        feature_list = []
        
        # Price features
        if 'close' in df.columns:
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)
        
        # Technical indicators
        indicator_cols = [
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'ema_9', 'ema_21', 'ema_50', 'ema_200',
            'sma_20', 'sma_50',
            'bb_high', 'bb_low', 'bb_mid',
            'atr', 'momentum', 'adx', 'stoch_k', 'stoch_d', 'williams_r',
            'volume_sma'
        ]
        
        for col in indicator_cols:
            if col in df.columns:
                feature_list.append(col)
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            feature_list.append('volume_ratio')
        
        # Create target (next period price direction)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Remove NaN rows
        feature_df = df[feature_list + ['target']].dropna()
        
        if len(feature_df) < 20:
            logger.warning(" Insufficient data after feature preparation")
            return pd.DataFrame(), pd.Series(dtype=float)
        
        X = feature_df[feature_list]
        y = feature_df['target']
        
        self.feature_columns = feature_list
        
        return X, y
    
    def train_ml_models(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Train RandomForest and XGBoost models
        
        Args:
            df: Training data
            symbol: Symbol name for model saving
            
        Returns:
            True if training successful
        """
        try:
            X, y = self.prepare_features(df)
            
            if X.empty or len(X) < 50:
                logger.warning(" Insufficient data for ML training")
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f" Training ML models on {len(X_train)} samples...")
            
            # Train RandomForest
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.rf_model.fit(X_train, y_train)
            rf_score = self.rf_model.score(X_test, y_test)
            logger.info(f" RandomForest trained - Accuracy: {rf_score:.3f}")
            
            # Train XGBoost
            if not XGBOOST_AVAILABLE:
                logger.warning("XGBoost not available, skipping XGBoost model training")
            else:
                self.xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                self.xgb_model.fit(X_train, y_train)
                xgb_score = self.xgb_model.score(X_test, y_test)
                logger.info(f"XGBoost trained - Accuracy: {xgb_score:.3f}")
            
            # Save models
            joblib.dump(self.rf_model, MODELS_DIR / f"rf_{symbol}.pkl")
            joblib.dump(self.xgb_model, MODELS_DIR / f"xgb_{symbol}.pkl")
            joblib.dump(self.scaler, MODELS_DIR / f"scaler_{symbol}.pkl")
            with open(MODELS_DIR / f"features_{symbol}.pkl", 'wb') as f:
                pickle.dump(self.feature_columns, f)
            
            logger.info(" ML models saved successfully")
            return True
            
        except Exception as e:
            logger.error(f" Error training ML models: {e}")
            return False
    
    def train_lstm(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Train LSTM model for time series prediction
        
        Args:
            df: Training data
            symbol: Symbol name for model saving
            
        Returns:
            True if training successful
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, cannot train LSTM model")
            return False
        
        try:
            if df.empty or len(df) < 100:
                logger.warning(" Insufficient data for LSTM training")
                return False
            
            # Prepare features
            X, y = self.prepare_features(df)
            
            if X.empty or len(X) < self.sequence_length + 10:
                logger.warning(" Insufficient data for LSTM sequences")
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Create sequences
            X_seq, y_seq = [], []
            for i in range(self.sequence_length, len(X_scaled)):
                X_seq.append(X_scaled[i-self.sequence_length:i])
                y_seq.append(y.iloc[i])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            # Split data
            split_idx = int(len(X_seq) * 0.8)
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
            
            logger.info(f" Training LSTM on {len(X_train)} sequences...")
            
            # Build LSTM model
            self.lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(self.sequence_length, X_scaled.shape[1])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1, activation='sigmoid')
            ])
            
            self.lstm_model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            self.lstm_model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=0
            )
            
            # Evaluate
            test_loss, test_acc = self.lstm_model.evaluate(X_test, y_test, verbose=0)
            logger.info(f" LSTM trained - Accuracy: {test_acc:.3f}")
            
            # Save model
            self.lstm_model.save(MODELS_DIR / f"lstm_{symbol}.h5")
            
            return True
            
        except Exception as e:
            logger.error(f" Error training LSTM: {e}")
            return False
    
    def predict_ml(self, df: pd.DataFrame) -> Dict:
        """
        Get predictions from ML models
        
        Args:
            df: Recent market data
            
        Returns:
            Dictionary with predictions and confidence
        """
        try:
            X, _ = self.prepare_features(df)
            
            if X.empty or len(X) < 1:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'No data'}
            
            # Get latest features
            latest_X = X.iloc[-1:].values
            latest_X_scaled = self.scaler.transform(latest_X)
            
            predictions = []
            confidences = []
            
            # RandomForest prediction
            if self.rf_model:
                rf_pred = self.rf_model.predict(latest_X_scaled)[0]
                rf_proba = self.rf_model.predict_proba(latest_X_scaled)[0]
                predictions.append(rf_pred)
                confidences.append(max(rf_proba))
            
            # XGBoost prediction
            if self.xgb_model:
                xgb_pred = self.xgb_model.predict(latest_X_scaled)[0]
                xgb_proba = self.xgb_model.predict_proba(latest_X_scaled)[0]
                predictions.append(xgb_pred)
                confidences.append(max(xgb_proba))
            
            if not predictions:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'Models not trained'}
            
            # Aggregate predictions
            avg_pred = np.mean(predictions)
            avg_confidence = np.mean(confidences)
            
            action = 'buy' if avg_pred > 0.5 else 'sell' if avg_pred < 0.5 else 'hold'
            
            return {
                'action': action,
                'confidence': float(avg_confidence),
                'reason': f'ML prediction: {action} (confidence: {avg_confidence:.2f})',
                'ml_score': float(avg_pred)
            }
            
        except Exception as e:
            logger.error(f" Error in ML prediction: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': str(e)}
    
    def predict_lstm(self, df: pd.DataFrame) -> Dict:
        """
        Get prediction from LSTM model
        
        Args:
            df: Recent market data
            
        Returns:
            Dictionary with prediction and confidence
        """
        try:
            if self.lstm_model is None:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'LSTM not trained'}
            
            X, _ = self.prepare_features(df)
            
            if len(X) < self.sequence_length:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'Insufficient data'}
            
            # Get sequence
            X_scaled = self.scaler.transform(X)
            X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Predict
            prediction = self.lstm_model.predict(X_seq, verbose=0)[0][0]
            confidence = abs(prediction - 0.5) * 2  # Convert to 0-1 range
            
            action = 'buy' if prediction > 0.5 else 'sell' if prediction < 0.5 else 'hold'
            
            return {
                'action': action,
                'confidence': float(confidence),
                'reason': f'LSTM prediction: {action} (score: {prediction:.3f})',
                'lstm_score': float(prediction)
            }
            
        except Exception as e:
            logger.error(f" Error in LSTM prediction: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': str(e)}
    
    def load_models(self, symbol: str) -> bool:
        """
        Load saved models
        
        Args:
            symbol: Symbol name
            
        Returns:
            True if models loaded successfully
        """
        try:
            # Load ML models
            rf_path = MODELS_DIR / f"rf_{symbol}.pkl"
            xgb_path = MODELS_DIR / f"xgb_{symbol}.pkl"
            scaler_path = MODELS_DIR / f"scaler_{symbol}.pkl"
            features_path = MODELS_DIR / f"features_{symbol}.pkl"
            
            if rf_path.exists():
                self.rf_model = joblib.load(rf_path)
                logger.info(" RandomForest model loaded")
            
            if xgb_path.exists():
                self.xgb_model = joblib.load(xgb_path)
                logger.info(" XGBoost model loaded")
            
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(" Scaler loaded")
            
            if features_path.exists():
                with open(features_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)
                logger.info(" Feature columns loaded")
            
            # Load LSTM
            lstm_path = MODELS_DIR / f"lstm_{symbol}.h5"
            if lstm_path.exists():
                self.lstm_model = load_model(lstm_path)
                logger.info(" LSTM model loaded")
            
            return True
            
        except Exception as e:
            logger.error(f" Error loading models: {e}")
            return False

