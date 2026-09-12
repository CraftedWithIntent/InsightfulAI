"""
DEPRECATED: This setup.py is kept for backward compatibility only.
Use pyproject.toml for package configuration (PEP 517/518 standard).

To install in development mode:
    pip install -e .

This will read configuration from pyproject.toml.
"""
from setuptools import setup

# Configuration is now in pyproject.toml
# setuptools will read it automatically
if __name__ == "__main__":
    setup()
