# Logging Setup Guide for InsightfulAI

## Overview

This guide provides instructions on configuring logging for Python projects. Logging helps track application events, errors, and other important data, which is essential for debugging and monitoring. In this guide, you’ll learn how to set up basic logging, customize log formatting, and output logs to files.

---

### Step 1: Basic Logging Setup

Python’s built-in `logging` module supports various logging levels that determine the severity of logged messages. Here’s how to set up basic logging with different logging levels:

1. **Import the logging module**:
   ```python
   import logging
   ```

2. **Configure logging**:
   Set the logging level and specify where logs will be displayed (console or file).
   ```python
   logging.basicConfig(level=logging.INFO)
   ```

3. **Log messages**:
   Use the following functions to log messages at different severity levels:
   ```python
   logging.debug("Debugging information")
   logging.info("General information")
   logging.warning("Warning about potential issues")
   logging.error("An error occurred")
   logging.critical("Critical error requiring immediate attention")
   ```

---

### Step 2: Customize Logging Output

Customize the log output format for more informative logging. Add timestamps, log levels, and message content to enhance readability.

1. **Set up a custom format**:
   ```python
   logging.basicConfig(
       level=logging.INFO,
       format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
       datefmt="%Y-%m-%d %H:%M:%S"
   )
   ```

2. **Format placeholders**:
   - `%(asctime)s`: Timestamp of the log.
   - `%(name)s`: Logger name.
   - `%(levelname)s`: Logging level (e.g., INFO, ERROR).
   - `%(message)s`: The actual log message.

3. **Example Output**:
   ```plaintext
   2024-11-09 14:32:45 - root - INFO - General information
   ```

---

### Step 3: File-based Logging

To save logs to a file, specify a `filename` in the `basicConfig`. File-based logging is useful for persistent logging, as log data can be retained across application runs.

1. **Set up logging to output to a file**:
   ```python
   logging.basicConfig(
       level=logging.INFO,
       format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
       datefmt="%Y-%m-%d %H:%M:%S",
       filename="application.log",
       filemode="w"  # 'w' for overwrite, 'a' for append
   )
   ```

2. **Rotate Logs (Optional)**:
   For long-running applications, log rotation can prevent log files from growing too large. Use `RotatingFileHandler` for automatic rotation based on file size.

   ```python
   from logging.handlers import RotatingFileHandler

   handler = RotatingFileHandler("application.log", maxBytes=5000, backupCount=3)
   logging.getLogger().addHandler(handler)
   ```

   - `maxBytes`: Maximum file size in bytes before rotation.
   - `backupCount`: Number of backup files to retain.

---

### Step 4: Using Logger Instances

Instead of using the root logger, create named logger instances to categorize logs by module or functionality. This approach improves organization in larger applications.

1. **Create a custom logger**:
   ```python
   logger = logging.getLogger("InsightfulAI")
   ```

2. **Configure the logger**:
   Set the log level and format for this specific logger.
   ```python
   handler = logging.StreamHandler()
   handler.setLevel(logging.INFO)
   formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
   handler.setFormatter(formatter)
   logger.addHandler(handler)
   logger.setLevel(logging.INFO)
   ```

3. **Use the logger**:
   ```python
   logger.info("InsightfulAI initialized successfully.")
   logger.error("An error occurred in module X.")
   ```

---

### Example Code: Configuring Logging in a Python Script

```python
import logging
from logging.handlers import RotatingFileHandler

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="application.log",
    filemode="w"
)

# Custom logger instance
logger = logging.getLogger("InsightfulAI")
handler = RotatingFileHandler("application.log", maxBytes=5000, backupCount=3)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Logging examples
logger.info("Application has started.")
logger.debug("Debugging message for InsightfulAI.")
logger.error("Error encountered in feature extraction module.")
```

---

### Additional Tips

- **Log Levels**: Set log levels to capture the right amount of information (e.g., `INFO` for general usage, `DEBUG` for in-depth diagnostics).
- **Log Rotation**: For high-traffic applications, rotate logs to manage file size.
- **Error Handling**: Use `try...except` blocks with `logger.error` to capture stack traces in logs.

---

### Summary

This guide covers basic logging setup, customized formatting, file-based logging, and advanced options like named loggers and log rotation. With logging configured, you’ll have improved insight into application events and errors, supporting better debugging and monitoring in InsightfulAI.

For more details on Python logging, refer to the [official Python logging documentation](https://docs.python.org/3/library/logging.html).