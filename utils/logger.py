"""Logging utilities"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class Logger:
    """Custom logger for the framework"""
    
    def __init__(self, name: str, log_dir: Optional[str] = None, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"{name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(console_format)
            self.logger.addHandler(file_handler)
            self.log_file = log_file
        else:
            self.log_file = None
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)


class MetricsLogger:
    """Logger for training metrics"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metrics_file = self.log_dir / f"metrics_{timestamp}.jsonl"
        self.metrics = []
    
    def log_metrics(self, step: int, metrics: dict):
        metrics_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics.append(metrics_entry)
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics_entry) + '\n')
    
    def get_metrics(self):
        return self.metrics


def get_logger(name: str, log_dir: Optional[str] = None) -> Logger:
    """Get or create a logger"""
    return Logger(name, log_dir)