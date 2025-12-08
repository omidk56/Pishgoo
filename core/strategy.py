"""
Trading Strategy Engine - Hybrid AI Strategy
"""

from typing import Dict, Optional
from core.data_fetcher import DataFetcher
from core.ai_model import AIModel
from core.prophet_model import ProphetForecaster
from core.risk_manager import RiskManager
from utils.indicators import TechnicalIndicators
from utils.logger import setup_logger

logger = setup_logger(__name__)


class HybridAIStrategy:
    """Hybrid AI trading strategy combining ML, LSTM, Prophet, and technical indicators"""
    
    def __init__(self, config: Dict, data_fetcher: DataFetcher, risk_manager: RiskManager):
        """
        Initialize hybrid AI strategy
        
        Args:
            config: Configuration dictionary
            data_fetcher: Data fetcher instance
            risk_manager: Risk manager instance
        """
        self.config = config
        self.data_fetcher = data_fetcher
        self.risk_manager = risk_manager
        self.ai_model = AIModel(config.get('ai', {}))
        self.prophet_model = ProphetForecaster(config.get('prophet', {}))
        self.indicators = TechnicalIndicators()
        self.confidence_threshold = config.get('ai', {}).get('confidence_threshold', 0.7)
        logger.info(" Hybrid AI Strategy initialized")
    
    def generate_signal(self, symbol: str) -> Dict:
        """
        Generate trading signal using all AI models and indicators
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with action, confidence, and reasoning
        """
        try:
            # Fetch market data
            df = self.data_fetcher.get_market_data(symbol, include_indicators=True)
            
            if df is None or df.empty:
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'reason': 'No market data available'
                }
            
            signals = []
            confidences = []
            reasons = []
            
            # 1. Technical Indicators Signal
            indicator_signal = self.indicators.get_signal_strength(df)
            signals.append(indicator_signal.get('action', 'hold'))
            confidences.append(indicator_signal.get('confidence', 0.0))
            reasons.append(f"Technical indicators: {indicator_signal.get('action', 'hold')}")
            
            # 2. ML Models Prediction
            if self.config.get('ai', {}).get('enabled', True):
                try:
                    # Try to load models if not loaded
                    if self.ai_model.rf_model is None:
                        self.ai_model.load_models(symbol)
                    
                    ml_prediction = self.ai_model.predict_ml(df)
                    if ml_prediction['confidence'] > 0:
                        signals.append(ml_prediction['action'])
                        confidences.append(ml_prediction['confidence'])
                        reasons.append(ml_prediction['reason'])
                    
                    # LSTM Prediction
                    if 'lstm' in self.config.get('ai', {}).get('models', []):
                        lstm_prediction = self.ai_model.predict_lstm(df)
                        if lstm_prediction['confidence'] > 0:
                            signals.append(lstm_prediction['action'])
                            confidences.append(lstm_prediction['confidence'])
                            reasons.append(lstm_prediction['reason'])
                except Exception as e:
                    logger.warning(f" AI prediction error: {e}")
            
            # 3. Prophet Forecasting
            if self.config.get('prophet', {}).get('enabled', True):
                try:
                    # Try to load Prophet model
                    if self.prophet_model.model is None:
                        self.prophet_model.load_model(symbol)
                    
                    if self.prophet_model.model is not None:
                        prophet_forecast = self.prophet_model.forecast()
                        prophet_action = prophet_forecast.get('direction', 'hold')
                        
                        if prophet_action == 'up':
                            signals.append('buy')
                        elif prophet_action == 'down':
                            signals.append('sell')
                        else:
                            signals.append('hold')
                        
                        confidences.append(prophet_forecast.get('confidence', 0.0))
                        reasons.append(f"Prophet: {prophet_forecast.get('trend', 'neutral')} trend")
                except Exception as e:
                    logger.warning(f" Prophet prediction error: {e}")
            
            # Aggregate signals
            if not signals:
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'reason': 'No signals generated'
                }
            
            # Count votes
            buy_votes = signals.count('buy')
            sell_votes = signals.count('sell')
            hold_votes = signals.count('hold')
            total_votes = len(signals)
            
            # Calculate weighted confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Determine action based on consensus
            if buy_votes > sell_votes and buy_votes > hold_votes:
                action = 'buy'
                signal_strength = buy_votes / total_votes
            elif sell_votes > buy_votes and sell_votes > hold_votes:
                action = 'sell'
                signal_strength = sell_votes / total_votes
            else:
                action = 'hold'
                signal_strength = 0.0
            
            # Final confidence = weighted average * signal strength
            final_confidence = avg_confidence * signal_strength
            
            # Check if confidence meets threshold
            if final_confidence < self.confidence_threshold:
                action = 'hold'
                reasons.append(f"Confidence {final_confidence:.2f} below threshold {self.confidence_threshold}")
            
            reason = " | ".join(reasons)
            
            logger.info(f" Signal generated for {symbol}: {action} (confidence: {final_confidence:.2f})")
            
            return {
                'action': action,
                'confidence': float(final_confidence),
                'reason': reason,
                'signals': {
                    'buy': buy_votes,
                    'sell': sell_votes,
                    'hold': hold_votes,
                    'total': total_votes
                },
                'details': {
                    'technical': indicator_signal,
                    'ml': ml_prediction if 'ml_prediction' in locals() else {},
                    'prophet': prophet_forecast if 'prophet_forecast' in locals() else {}
                }
            }
            
        except Exception as e:
            logger.error(f" Error generating signal: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reason': f'Error: {str(e)}'
            }
    
    def should_execute_trade(self, signal: Dict, symbol: str) -> bool:
        """
        Determine if trade should be executed based on signal and risk management
        
        Args:
            signal: Trading signal dictionary
            symbol: Trading pair symbol
            
        Returns:
            True if trade should be executed
        """
        # Check confidence threshold
        if signal['confidence'] < self.confidence_threshold:
            logger.debug(f" Signal confidence {signal['confidence']:.2f} below threshold")
            return False
        
        # Check action
        if signal['action'] == 'hold':
            return False
        
        # Check risk management
        # (Additional checks can be added here)
        
        return True



