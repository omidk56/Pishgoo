"""
Backtesting System for Pishgoo
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from core.strategy import HybridAIStrategy
from core.data_fetcher import DataFetcher
from core.risk_manager import RiskManager
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Backtester:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, strategy: HybridAIStrategy, initial_balance: float = 100000000):
        """
        Initialize backtester
        
        Args:
            strategy: Strategy instance
            initial_balance: Initial balance in IRT
        """
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: List[Dict] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_balance]
        logger.info(f" Backtester initialized with balance: {initial_balance:,.0f} IRT")
    
    def run_backtest(self, df: pd.DataFrame, symbol: str, start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            df: Historical OHLCV data
            symbol: Trading pair symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Filter dates if provided
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            if df.empty or len(df) < 50:
                logger.warning(" Insufficient data for backtesting")
                return self._empty_results()
            
            logger.info(f" Running backtest on {len(df)} candles from {df.index[0]} to {df.index[-1]}")
            
            # Reset state
            self.balance = self.initial_balance
            self.positions = []
            self.trades = []
            self.equity_curve = [self.initial_balance]
            
            # Iterate through data
            for i in range(50, len(df)):  # Start from index 50 to have enough history
                current_data = df.iloc[:i+1]
                current_price = df.iloc[i]['close']
                current_time = df.index[i]
                
                # Close positions if stop loss or take profit hit
                self._check_exit_conditions(current_price, current_time)
                
                # Generate signal
                # Temporarily set data_fetcher data for strategy
                old_cache = self.strategy.data_fetcher._cache
                self.strategy.data_fetcher._cache = {f"{symbol}_1h": (0, current_data)}
                
                try:
                    signal = self.strategy.generate_signal(symbol)
                except:
                    signal = {'action': 'hold', 'confidence': 0.0}
                
                # Restore cache
                self.strategy.data_fetcher._cache = old_cache
                
                # Execute trade if signal is strong enough
                if self.strategy.should_execute_trade(signal, symbol):
                    self._execute_trade(signal, symbol, current_price, current_time)
                
                # Update equity curve
                self._update_equity(current_price)
            
            # Close all remaining positions
            final_price = df.iloc[-1]['close']
            for pos in self.positions:
                self._close_position(pos, final_price, df.index[-1])
            
            # Calculate metrics
            results = self._calculate_metrics(df)
            logger.info(f" Backtest completed. Total return: {results['total_return']:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f" Backtest error: {e}")
            return self._empty_results()
    
    def _execute_trade(self, signal: Dict, symbol: str, price: float, timestamp: pd.Timestamp):
        """Execute a trade"""
        if signal['action'] == 'hold':
            return
        
        # Calculate position size
        position_size = self.strategy.risk_manager.calculate_position_size(
            self.balance, price
        )
        
        if position_size <= 0:
            return
        
        # Check if we can afford the trade
        cost = position_size * price
        if cost > self.balance:
            return
        
        # Validate order
        validation = self.strategy.risk_manager.validate_order(
            signal['action'], position_size, price, self.balance
        )
        
        if not validation['valid']:
            return
        
        # Create position
        position = {
            'id': len(self.positions) + 1,
            'symbol': symbol,
            'side': signal['action'],
            'entry_price': price,
            'amount': position_size,
            'entry_time': timestamp,
            'stop_loss': self.strategy.risk_manager.calculate_stop_loss_price(price, signal['action']),
            'take_profit': self.strategy.risk_manager.calculate_take_profit_price(price, signal['action']),
            'signal': signal
        }
        
        self.positions.append(position)
        self.balance -= cost
        
        logger.debug(f" Position opened: {signal['action']} {position_size:.4f} {symbol} @ {price:,.0f}")
    
    def _check_exit_conditions(self, current_price: float, timestamp: pd.Timestamp):
        """Check stop loss and take profit conditions"""
        positions_to_close = []
        
        for pos in self.positions:
            triggered = False
            
            # Check stop loss
            if self.strategy.risk_manager.check_stop_loss(
                pos['entry_price'], current_price, pos['side']
            ):
                triggered = True
            
            # Check take profit
            if self.strategy.risk_manager.check_take_profit(
                pos['entry_price'], current_price, pos['side']
            ):
                triggered = True
            
            if triggered:
                positions_to_close.append(pos)
        
        for pos in positions_to_close:
            self._close_position(pos, current_price, timestamp)
    
    def _close_position(self, position: Dict, exit_price: float, timestamp: pd.Timestamp):
        """Close a position"""
        if position['side'] == 'buy':
            pnl = (exit_price - position['entry_price']) * position['amount']
        else:  # sell
            pnl = (position['entry_price'] - exit_price) * position['amount']
        
        self.balance += (position['amount'] * exit_price) + pnl
        
        trade = {
            'id': position['id'],
            'symbol': position['symbol'],
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'amount': position['amount'],
            'pnl': pnl,
            'pnl_pct': (pnl / (position['entry_price'] * position['amount'])) * 100,
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'duration': (timestamp - position['entry_time']).total_seconds() / 3600,  # hours
            'signal': position['signal']
        }
        
        self.trades.append(trade)
        self.positions.remove(position)
        
        logger.debug(f" Position closed: {trade['side']} {trade['amount']:.4f} @ {exit_price:,.0f} | PnL: {pnl:,.0f} ({trade['pnl_pct']:.2f}%)")
    
    def _update_equity(self, current_price: float):
        """Update equity curve"""
        equity = self.balance
        
        # Add unrealized PnL from open positions
        for pos in self.positions:
            if pos['side'] == 'buy':
                equity += (current_price - pos['entry_price']) * pos['amount']
            else:  # sell
                equity += (pos['entry_price'] - current_price) * pos['amount']
        
        self.equity_curve.append(equity)
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate backtest performance metrics"""
        if not self.trades:
            return self._empty_results()
        
        trades_df = pd.DataFrame(self.trades)
        
        # Total return
        total_return = ((self.equity_curve[-1] - self.initial_balance) / self.initial_balance) * 100
        
        # Win rate
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        total_trades = len(trades_df)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Average win/loss
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
        profit_factor = (avg_win * winning_trades) / (avg_loss * (total_trades - winning_trades)) if avg_loss > 0 and (total_trades - winning_trades) > 0 else 0
        
        # Max drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = ((equity_array - running_max) / running_max) * 100
        max_drawdown = abs(drawdown.min())
        
        # Sharpe ratio (simplified)
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Total PnL
        total_pnl = trades_df['pnl'].sum()
        
        return {
            'total_return': float(total_return),
            'total_pnl': float(total_pnl),
            'initial_balance': float(self.initial_balance),
            'final_balance': float(self.equity_curve[-1]),
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'losing_trades': int(total_trades - winning_trades),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': float(sharpe_ratio),
            'equity_curve': self.equity_curve,
            'trades': trades_df.to_dict('records')
        }
    
    def _empty_results(self) -> Dict:
        """Return empty results structure"""
        return {
            'total_return': 0.0,
            'total_pnl': 0.0,
            'initial_balance': float(self.initial_balance),
            'final_balance': float(self.initial_balance),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'equity_curve': [self.initial_balance],
            'trades': []
        }



