"""
InsightfulAI - Retry with Exponential Backoff Decorator
=======================================================

Project: InsightfulAI
Repository: https://github.com/CraftedWithIntent/InsightfulAI
Author: Philip Thomas
Date: 2024-11-13

Description:
This module provides a decorator function, `retry_exponential_backoff`, which applies retry logic to any function. 
When an exception is raised during the function's execution, the decorator retries up to a specified number of attempts, 
with an exponentially increasing backoff time (2^attempt seconds) between retries. If all retries fail, an error 
is logged, and the exception is raised.

Usage:
Apply `@retry_exponential_backoff` to functions where transient errors may occur (e.g., network calls, database queries).
This will log each retry attempt, wait with exponential backoff, and ultimately raise an error if retries are exhausted.

Parameters:
- `func` (callable): The function to wrap with retry logic.

Example:
```python
@retry_exponential_backoff
def risky_operation():
    # Code that may raise an exception
    pass

"""

from functools import wraps
import time
import logging

def retry_exponential_backoff(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Attempt {attempt} for {func.__name__} failed with error: {e}")
                
                # If max retries reached, log error and raise exception
                if attempt == max_retries:
                    logging.error(f"{func.__name__} failed after {max_retries} attempts. Raising exception.")
                    raise
                
                # Calculate and log backoff wait time
                wait_time = 2 ** attempt
                logging.info(f"Retrying {func.__name__} in {wait_time} seconds (attempt {attempt + 1} of {max_retries})...")
                time.sleep(wait_time)
    
    return wrapper