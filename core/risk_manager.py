"""
Risk Management System
"""

from typing import Dict, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RiskManager:
    """Manage trading risk and position sizing"""
    
    def __init__(self, risk_config: Dict):
        """
        Initialize risk manager
        
        Args:
            risk_config: Risk configuration dictionary
        """
        self.stop_loss = risk_config.get('stop_loss', 0.03)  # 3% default
        self.take_profit = risk_config.get('take_profit', 0.05)  # 5% default
        self.max_position_size = risk_config.get('max_position_size', 0.2)  # 20% of balance
        logger.info(f" Risk manager initialized - SL: {self.stop_loss*100}%, TP: {self.take_profit*100}%")
    
    def calculate_position_size(self, balance: float, price: float, risk_percent: float = None) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            balance: Available balance
            price: Entry price
            risk_percent: Risk percentage (default: max_position_size)
            
        Returns:
            Position size in base currency
        """
        risk_percent = risk_percent or self.max_position_size
        max_amount = balance * risk_percent
        
        # Ensure we don't exceed maximum position size
        position_size = min(max_amount, balance * self.max_position_size)
        
        logger.debug(f" Position size calculated: {position_size:.2f} (max: {balance * self.max_position_size:.2f})")
        return position_size
    
    def validate_order(self, side: str, amount: float, price: float, balance: float) -> Dict[str, bool]:
        """
        Validate order before execution
        
        Args:
            side: 'buy' or 'sell'
            amount: Order amount
            price: Order price
            balance: Available balance
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'reason': 'Order validated'
        }
        
        # Check balance
        required_balance = amount * price if side == 'buy' else amount
        
        if required_balance > balance:
            result['valid'] = False
            result['reason'] = f'Insufficient balance. Required: {required_balance:.2f}, Available: {balance:.2f}'
            logger.warning(f" {result['reason']}")
            return result
        
        # Check position size
        max_position = balance * self.max_position_size
        position_value = amount * price
        
        if position_value > max_position:
            result['valid'] = False
            result['reason'] = f'Position size exceeds maximum. Position: {position_value:.2f}, Max: {max_position:.2f}'
            logger.warning(f" {result['reason']}")
            return result
        
        logger.info(f" Order validated: {side} {amount} @ {price}")
        return result
    
    def calculate_stop_loss_price(self, entry_price: float, side: str) -> float:
        """
        Calculate stop loss price
        
        Args:
            entry_price: Entry price
            side: 'buy' or 'sell'
            
        Returns:
            Stop loss price
        """
        if side == 'buy':
            stop_loss_price = entry_price * (1 - self.stop_loss)
        else:  # sell
            stop_loss_price = entry_price * (1 + self.stop_loss)
        
        return stop_loss_price
    
    def calculate_take_profit_price(self, entry_price: float, side: str) -> float:
        """
        Calculate take profit price
        
        Args:
            entry_price: Entry price
            side: 'buy' or 'sell'
            
        Returns:
            Take profit price
        """
        if side == 'buy':
            take_profit_price = entry_price * (1 + self.take_profit)
        else:  # sell
            take_profit_price = entry_price * (1 - self.take_profit)
        
        return take_profit_price
    
    def check_stop_loss(self, entry_price: float, current_price: float, side: str) -> bool:
        """
        Check if stop loss should be triggered
        
        Args:
            entry_price: Entry price
            current_price: Current market price
            side: 'buy' or 'sell'
            
        Returns:
            True if stop loss triggered
        """
        stop_loss_price = self.calculate_stop_loss_price(entry_price, side)
        
        if side == 'buy':
            triggered = current_price <= stop_loss_price
        else:  # sell
            triggered = current_price >= stop_loss_price
        
        if triggered:
            logger.warning(f" Stop loss triggered! Entry: {entry_price}, Current: {current_price}, SL: {stop_loss_price}")
        
        return triggered
    
    def check_take_profit(self, entry_price: float, current_price: float, side: str) -> bool:
        """
        Check if take profit should be triggered
        
        Args:
            entry_price: Entry price
            current_price: Current market price
            side: 'buy' or 'sell'
            
        Returns:
            True if take profit triggered
        """
        take_profit_price = self.calculate_take_profit_price(entry_price, side)
        
        if side == 'buy':
            triggered = current_price >= take_profit_price
        else:  # sell
            triggered = current_price <= take_profit_price
        
        if triggered:
            logger.info(f" Take profit triggered! Entry: {entry_price}, Current: {current_price}, TP: {take_profit_price}")
        
        return triggered



