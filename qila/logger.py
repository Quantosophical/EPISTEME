"""
Episteme Structured Logger
==========================
Centralized logging configuration for all QILA modules.
Logs to both console (colored) and file (episteme.log).
"""

import os
import sys
import logging
from pathlib import Path

# Determine log file location (project root)
_project_root = Path(__file__).resolve().parent.parent
_log_file = _project_root / "episteme.log"


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger with the given name.
    
    Usage:
        from logger import get_logger
        log = get_logger(__name__)
        log.info("Something happened")
        log.warning("Retry needed")
        log.error("Failed permanently")
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Console handler — INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # File handler — DEBUG and above (full detail)
    try:
        file_handler = logging.FileHandler(_log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(funcName)s:%(lineno)d │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    except (OSError, PermissionError):
        pass  # Can't write log file — console only
    
    return logger
