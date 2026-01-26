import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup structured logger with file and console output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_assumption(logger: logging.Logger, assumption: str, value: str = None):
    """Log explicit assumptions per spec N."""
    msg = f"ASSUMPTION: {assumption}"
    if value:
        msg += f" = {value}"
    logger.warning(msg)

def log_mock_mode(logger: logging.Logger):
    """Per spec L: clearly log when in mock mode."""
    logger.warning("=" * 50)
    logger.warning("MOCK MODE ACTIVE - Using synthetic data")
    logger.warning("=" * 50)
