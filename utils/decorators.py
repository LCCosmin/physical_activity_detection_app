from typing import Callable, Any
from datetime import datetime
import logging

def benchmark(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        run_time = start_time - end_time
        logging.info(f"Execution of function {func.__name__} took {run_time}.")
        return result
    
    return wrapper
