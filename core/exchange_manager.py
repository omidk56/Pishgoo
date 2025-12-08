"""
Exchange Manager - Unified interface for multiple exchanges
"""

from typing import Dict, Optional
from exchanges.base import BaseExchange
from exchanges.nobitex import NobitexExchange
from exchanges.wallex import WallexExchange
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ExchangeManager:
    """Manage multiple exchanges with unified interface"""
    
    def __init__(self, config: Dict):
        """
        Initialize exchange manager
        
        Args:
            config: Configuration dictionary with exchange settings
        """
        self.config = config
        self.exchange_type = config.get('exchange', 'nobitex').lower()
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.exchange: Optional[BaseExchange] = None
        
        self._initialize_exchange()
    
    def _initialize_exchange(self) -> None:
        """Initialize the selected exchange"""
        try:
            if self.exchange_type == 'nobitex':
                self.exchange = NobitexExchange(self.api_key, self.api_secret)
                logger.info("Nobitex exchange manager initialized")
            elif self.exchange_type == 'wallex':
                self.exchange = WallexExchange(self.api_key, self.api_secret)
                logger.info("Wallex exchange manager initialized")
            else:
                logger.error(f"Unsupported exchange: {self.exchange_type}")
                raise ValueError(f"Unsupported exchange: {self.exchange_type}")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        if not self.exchange:
            logger.error("Exchange not initialized")
            return {}
        return self.exchange.get_balance()
    
    def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100):
        """Get OHLCV data"""
        if not self.exchange:
            logger.error("Exchange not initialized")
            return None
        return self.exchange.get_ohlcv(symbol, timeframe, limit)
    
    def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None):
        """Place an order"""
        if not self.exchange:
            logger.error("Exchange not initialized")
            return None
        return self.exchange.place_order(symbol, side, amount, price)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.exchange:
            logger.error("Exchange not initialized")
            return False
        return self.exchange.cancel_order(order_id)
    
    def get_open_orders(self, symbol: Optional[str] = None):
        """Get open orders"""
        if not self.exchange:
            logger.error("Exchange not initialized")
            return []
        return self.exchange.get_open_orders(symbol)
    
    def get_ticker(self, symbol: str):
        """Get ticker information"""
        if not self.exchange:
            logger.error("Exchange not initialized")
            return {}
        return self.exchange.get_ticker(symbol)

