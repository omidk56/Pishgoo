"""
Configuration management for Pishgoo
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)

CONFIG_DIR = Path(__file__).parent
USER_CONFIG_PATH = CONFIG_DIR / "user_config.json"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.json"


def get_default_config() -> Dict[str, Any]:
    """Return default configuration"""
    return {
        "exchange": "nobitex",
        "api_key": "",
        "api_secret": "",
        "pairs": ["BTCIRT", "ETHIRT"],
        "amount_per_trade": 5000000,
        "risk": {
            "stop_loss": 0.03,
            "take_profit": 0.05,
            "max_position_size": 0.2
        },
        "ai": {
            "enabled": True,
            "models": ["ml", "lstm", "prophet"],
            "confidence_threshold": 0.7
        },
        "prophet": {
            "enabled": True,
            "forecast_periods": 24,
            "seasonality_mode": "multiplicative"
        },
        "trading": {
            "enabled": False,
            "strategy": "hybrid_ai",
            "timeframe": "1h"
        },
        "dashboard": {
            "password": "pishgoo123",
            "port": 8501,
            "host": "0.0.0.0",
            "language": "en"
        }
    }


def load_config() -> Optional[Dict[str, Any]]:
    """Load user configuration or create default"""
    try:
        if USER_CONFIG_PATH.exists():
            with open(USER_CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {USER_CONFIG_PATH}")
                return config
        else:
            # Create default config
            config = get_default_config()
            save_config(config)
            logger.info(f"Created default configuration at {USER_CONFIG_PATH}")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None


def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to file"""
    try:
        CONFIG_DIR.mkdir(exist_ok=True)
        with open(USER_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logger.info(f"Configuration saved to {USER_CONFIG_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False


def update_config(updates: Dict[str, Any]) -> bool:
    """Update configuration with new values"""
    config = load_config()
    if not config:
        return False
    
    # Deep merge updates
    for key, value in updates.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            config[key].update(value)
        else:
            config[key] = value
    
    return save_config(config)

