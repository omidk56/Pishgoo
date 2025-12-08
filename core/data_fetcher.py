"""
Data Fetcher - Fetch and prepare market data
"""

import pandas as pd
import time
from typing import Dict, Optional
from core.exchange_manager import ExchangeManager
from utils.indicators import TechnicalIndicators
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DataFetcher:
    """Fetch and prepare market data with indicators"""
    
    def __init__(self, exchange_manager: ExchangeManager):
        """
        Initialize data fetcher
        
        Args:
            exchange_manager: Exchange manager instance
        """
        self.exchange_manager = exchange_manager
        self.indicators = TechnicalIndicators()
        self._cache: Dict[str, Dict] = {}
        logger.info("Data fetcher initialized")
    
    def get_market_data(self, symbol: str, timeframe: str = "1h", 
                       limit: int = 200, include_indicators: bool = True) -> Optional[pd.DataFrame]:
        """
        Get market data with indicators
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (1h, 4h, 1d)
            limit: Number of candles
            include_indicators: Whether to calculate indicators
            
        Returns:
            DataFrame with OHLCV and indicators
        """
        try:
            # Check cache
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self._cache:
                cached_time, cached_data = self._cache[cache_key]
                if time.time() - cached_time < 60:  # 1 minute cache
                    logger.debug(f"Using cached data for {symbol}")
                    return cached_data.copy()
            
            # Fetch from exchange
            df = self.exchange_manager.get_ohlcv(symbol, timeframe, limit)
            
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return None
            
            # Ensure correct column names
            if 'open' not in df.columns:
                df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Calculate indicators
            if include_indicators:
                df = self.indicators.calculate_all(df)
            
            # Cache the data
            self._cache[cache_key] = (time.time(), df.copy())
            
            logger.info(f"Market data fetched: {symbol} ({len(df)} candles)")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            ticker = self.exchange_manager.get_ticker(symbol)
            return ticker.get('last', 0.0)
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return 0.0
    
    def clear_cache(self) -> None:
        """Clear data cache"""
        self._cache.clear()
        logger.info("Cache cleared")

