import logging
import time
from functools import wraps
from rich.logging import RichHandler
from typing import Callable


def setup_logger(name: str) -> logging.Logger:
    """Setup a logger

    Parameters
    ----------
    name : str
        name of the logger

    Returns
    -------
    logging.Logger
        a new logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s", "%Y-%m-%d %H:%M:%S")
    handler = RichHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def setup_timer(logger: logging.Logger) -> Callable:
    """Create a decorator to measure elapsed time

    Parameters
    ----------
    logger : logging.Logger
        logger

    Returns
    -------
    Callable
        a new decorator
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start
            logger.info(f"{func.__name__} done in {elapsed_time:.2f} seconds")
            return result

        return wrapper

    return decorator
