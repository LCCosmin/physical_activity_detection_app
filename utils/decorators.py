from typing import Callable, Any
from time import time
import logging


def get_logger():
    logging.basicConfig(
        level = logging.INFO,
        format = '%(levelname)s:%(name)s: %(message)s'
    )

    logger = logging.getLogger("APP_LOGGER")
    return logger


def benchmark(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        run_time = end_time - start_time

        logger = get_logger()
        logger.info(f"Execution of function {str(func.__name__).upper()} took {run_time} seconds.")
        return result
    
    return wrapper
