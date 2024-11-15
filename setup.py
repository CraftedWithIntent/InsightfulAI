from setuptools import setup, find_packages

setup(
    name="InsightfulAI",                          # Package name
    version="0.3.0a1",                              # Initial version
    author="Philip Thomas",
    description="A simple ML package for classification and regression",
    long_description=open("README.md", encoding="utf-8").read(),  # Specify UTF-8 encoding
    long_description_content_type="text/markdown",
    packages=find_packages(),                     # Automatically find and include submodules
    install_requires=[
        "scikit-learn", 
        "numpy", 
        "opentelemetry-api", 
        "opentelemetry-sdk", 
        "opentelemetry-instrumentation", 
        "asyncio"
    ],   # Dependencies
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)