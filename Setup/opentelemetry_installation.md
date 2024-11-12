# OpenTelemetry Installation Guide for InsightfulAI

## Overview

This guide provides instructions on setting up **pip** (Python’s package installer) and installing **OpenTelemetry** for Python. OpenTelemetry enables tracing in InsightfulAI, improving observability and monitoring across projects.

---

### Step 1: Install pip

**pip** is the package manager for Python, used to install and manage Python packages.

1. **Check if pip is installed**:
   Open your terminal (or command prompt) and type:
   ```bash
   pip --version
   ```
   If pip is installed, you’ll see the version displayed (e.g., `pip 21.0.1`). If it’s not installed, follow the steps below to install it.

2. **Install pip**:
   - **For Python 3 (recommended)**:
     ```bash
     python3 -m ensurepip --upgrade
     ```
   - **On Windows**:
     Download [get-pip.py](https://bootstrap.pypa.io/get-pip.py) and then run:
     ```bash
     python get-pip.py
     ```

3. **Verify pip installation**:
   Confirm that pip is installed by checking the version:
   ```bash
   pip --version
   ```

---

### Step 2: Install OpenTelemetry Packages

Once pip is installed, use it to install the OpenTelemetry packages for Python.

1. **Install OpenTelemetry packages**:
   Run the following command to install the required OpenTelemetry packages:
   ```bash
   pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation
   ```

2. **Verify Installation**:
   Ensure the packages installed correctly by checking each one:
   ```bash
   pip show opentelemetry-api
   pip show opentelemetry-sdk
   pip show opentelemetry-instrumentation
   ```
   Each command should display information about the package if it’s installed.

---

### Step 3: Setting Up OpenTelemetry in Your Project

To begin using OpenTelemetry in your Python project, add the necessary import statements and configure a basic setup. Here’s an example with **ConsoleSpanExporter** to output trace data to the console.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Set up tracer provider
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure ConsoleSpanExporter to output trace data to the console
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Example of tracing a custom operation
with tracer.start_as_current_span("example_operation"):
    print("Tracing an example operation...")
```

In this setup:
- **TracerProvider** and **Tracer** are configured to create and track spans.
- **ConsoleSpanExporter** outputs trace data to the console, useful for local testing and debugging.

### Sample Console Output

With **ConsoleSpanExporter** enabled, your console will display trace data similar to this:

```plaintext
{
    "trace_id": "...",
    "span_id": "...",
    "name": "example_operation",
    "status": {
        "status_code": "OK"
    },
    "attributes": {
        "custom.example": "Tracing an example operation"
    }
}
```

---

### Additional Tips

- **Virtual Environments**: Use virtual environments to manage packages and dependencies in isolation.
  - Create a virtual environment:
    ```bash
    python3 -m venv venv
    ```
  - Activate the environment:
    - **On Windows**:
      ```bash
      .\venv\Scripts\activate
      ```
    - **On macOS/Linux**:
      ```bash
      source venv/bin/activate
      ```
  - Install packages within this environment to avoid conflicts with your main system.

- **Updating pip**: Keep pip updated to avoid compatibility issues.
  ```bash
  python -m pip install --upgrade pip
  ```

---

### Summary

By following this guide, you’ll have **pip** installed for package management and **OpenTelemetry** set up with **ConsoleSpanExporter** for tracing in your InsightfulAI projects. This setup enables essential observability features, improving monitoring and debugging capabilities.

Let me know if you need further customization!
```