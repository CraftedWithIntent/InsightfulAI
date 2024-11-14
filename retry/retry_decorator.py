"""
InsightfulAI - Retry with Exponential Backoff Decorator
=======================================================

Project: InsightfulAI
Repository: https://github.com/CraftedWithIntent/InsightfulAI
Author: Philip Thomas
Date: 2024-11-13

Description:
This module provides a decorator function, `retry_exponential_backoff`, that wraps any function with retry logic. 
If an exception occurs, the decorator will retry the function up to a maximum of three attempts. Each retry 
waits for an exponentially increasing backoff time (2^attempt seconds) before the next attempt. 
If all retries fail, an error is logged, and the exception is raised.

Usage:
Apply the `@retry_exponential_backoff` decorator to any function where transient errors are expected 
(e.g., network calls, API requests). This will log each attempt and wait before retrying in case of failure.

Parameters:
- `func` (callable): The function to be wrapped with retry logic.

Example:
```python
@retry_exponential_backoff
def risky_operation():
    # Function code that may raise an exception
    pass
"""

import time
import logging

def retry_exponential_backoff(func):
    def wrapper(*args, **kwargs):
        max_retries = 3
        attempts = 0
        while attempts < max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                attempts += 1
                logging.error(f"Attempt {attempts} failed with error: {e}")
                if attempts < max_retries:
                    wait_time = 2 ** attempts  # Exponential backoff
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error("Max retries reached.")
                    raise
    return wrapper