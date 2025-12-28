"""Configuration loader and manager"""
import yaml
from typing import Any, Dict
from pathlib import Path
import torch

class Config:
    """Configuration manager with dot notation access"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        self._parse_nested(config_dict)
    
    def _parse_nested(self, d: Dict[str, Any], prefix: str = ""):
        """Parse nested dictionary and create attributes"""
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self._config


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def get_device(config: Config) -> torch.device:
    """Get torch device from configuration"""
    device_str = config.system.device
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")