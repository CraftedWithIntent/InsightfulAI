import time
import logging

def retry_with_backoff(func):
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