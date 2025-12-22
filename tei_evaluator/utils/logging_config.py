"""
Logging configuration for TEI Evaluator.

This module provides centralized logging configuration for all evaluation dimensions.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and/or console handlers.

    Args:
        name: Logger name (typically __name__ of the calling module)
        log_file: Path to log file. If None, only console logging is used
        level: Logging level (default: INFO)
        console_output: Whether to output to console (default: True)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to allow reconfiguration
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_evaluation_logger(dimension: int, output_dir: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for a specific evaluation dimension.

    Args:
        dimension: Evaluation dimension number (0-4)
        output_dir: Output directory for log files. If None, logs to logs/ directory

    Returns:
        Configured logger for the dimension
    """
    logger_name = f"tei_evaluator.d{dimension}"

    if output_dir:
        log_file = Path(output_dir) / "logs" / f"d{dimension}_evaluation.log"
    else:
        log_file = Path("logs") / f"d{dimension}_evaluation.log"

    return setup_logger(logger_name, str(log_file))

