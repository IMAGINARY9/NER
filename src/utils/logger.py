"""
Logging module for the Named Entity Recognition project.
"""

import os
import logging
import logging.handlers
import datetime
from pathlib import Path


def setup_logging(log_dir='./logs', log_level=logging.INFO, log_to_console=True):
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level
        log_to_console: Whether to also log to console
    """
    # Create log directory if it doesn't exist
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir_path / f"ner_{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove all existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create a file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    root_logger.addHandler(file_handler)
    
    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Log initial message
    logging.info(f"Logging initialized. Log file: {log_filename}")
    
    return root_logger
