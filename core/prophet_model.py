"""
Prophet Forecasting Engine for Pishgoo
"""

import pandas as pd
import numpy as np

# Optional import for Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None
from typing import Dict, Optional, Tuple
from pathlib import Path
import pickle
from utils.logger import setup_logger

logger = setup_logger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class ProphetForecaster:
    """Prophet-based price forecasting"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Prophet forecaster
        
        Args:
            config: Configuration dictionary for Prophet
        """
        self.config = config or {}
        self.model: Optional[Prophet] = None
        self.forecast_periods = self.config.get('forecast_periods', 24)
        self.seasonality_mode = self.config.get('seasonality_mode', 'multiplicative')
        logger.info(" Prophet forecaster initialized")
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Prophet (needs 'ds' and 'y' columns)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Prophet format
        """
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df['close'].values
        })
        prophet_df = prophet_df.dropna()
        return prophet_df
    
    def train(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Train Prophet model on historical data
        
        Args:
            df: Historical OHLCV data
            symbol: Trading pair symbol (for model saving)
            
        Returns:
            True if training successful
        """
        try:
            if df.empty or len(df) < 50:
                logger.warning(f" Insufficient data for Prophet training: {len(df)} rows")
                return False
            
            prophet_df = self.prepare_data(df)
            
            if len(prophet_df) < 50:
                logger.warning(f" Insufficient Prophet data: {len(prophet_df)} rows")
                return False
            
            if not PROPHET_AVAILABLE:
                logger.warning("Prophet not available, cannot train Prophet model")
                return False
            
            # Initialize Prophet model
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            # Fit model
            logger.info(f" Training Prophet model on {len(prophet_df)} data points...")
            self.model.fit(prophet_df)
            
            # Save model
            model_path = MODELS_DIR / f"prophet_{symbol}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            logger.info(f" Prophet model trained and saved: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f" Error training Prophet model: {e}")
            return False
    
    def load_model(self, symbol: str) -> bool:
        """
        Load saved Prophet model
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            True if model loaded successfully
        """
        try:
            model_path = MODELS_DIR / f"prophet_{symbol}.pkl"
            if not model_path.exists():
                logger.warning(f" No saved model found for {symbol}")
                return False
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info(f" Prophet model loaded: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f" Error loading Prophet model: {e}")
            return False
    
    def forecast(self, periods: Optional[int] = None) -> Dict:
        """
        Generate forecast using trained model
        
        Args:
            periods: Number of periods to forecast (default: self.forecast_periods)
            
        Returns:
            Dictionary with forecast data and signals
        """
        if self.model is None:
            logger.error(" Prophet model not trained or loaded")
            return {
                'forecast_df': pd.DataFrame(),
                'direction': 'hold',
                'confidence': 0.0,
                'trend': 'neutral'
            }
        
        try:
            periods = periods or self.forecast_periods
            
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods)
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Get forecasted values (last N periods)
            forecasted = forecast.tail(periods)
            
            # Calculate direction and confidence
            current_price = forecast['yhat'].iloc[-periods-1] if len(forecast) > periods else forecast['yhat'].iloc[-1]
            future_prices = forecasted['yhat'].values
            
            # Trend analysis
            price_change = (future_prices[-1] - current_price) / current_price
            
            # Calculate confidence based on forecast uncertainty
            uncertainty = forecasted['yhat_upper'].values - forecasted['yhat_lower'].values
            avg_uncertainty = np.mean(uncertainty)
            price_range = max(future_prices) - min(future_prices)
            
            if price_range > 0:
                confidence = min(1.0, max(0.0, 1.0 - (avg_uncertainty / price_range)))
            else:
                confidence = 0.5
            
            # Determine direction
            if price_change > 0.02:  # 2% increase
                direction = 'up'
                trend = 'bullish'
            elif price_change < -0.02:  # 2% decrease
                direction = 'down'
                trend = 'bearish'
            else:
                direction = 'hold'
                trend = 'neutral'
            
            logger.info(f" Prophet forecast: {direction} trend, confidence: {confidence:.2f}")
            
            return {
                'forecast_df': forecasted[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                'direction': direction,
                'confidence': float(confidence),
                'trend': trend,
                'price_change_pct': float(price_change * 100),
                'current_price': float(current_price),
                'forecasted_price': float(future_prices[-1])
            }
            
        except Exception as e:
            logger.error(f" Error generating Prophet forecast: {e}")
            return {
                'forecast_df': pd.DataFrame(),
                'direction': 'hold',
                'confidence': 0.0,
                'trend': 'neutral'
            }
    
    def train_and_forecast(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Train model and generate forecast in one step
        
        Args:
            df: Historical OHLCV data
            symbol: Trading pair symbol
            
        Returns:
            Forecast dictionary
        """
        if self.train(df, symbol):
            return self.forecast()
        else:
            return {
                'forecast_df': pd.DataFrame(),
                'direction': 'hold',
                'confidence': 0.0,
                'trend': 'neutral'
            }

