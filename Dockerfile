# Multi-stage Dockerfile for InsightfulAI ML framework

FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy source (all Python modules and config)
COPY pyproject.toml setup.py ./
COPY models ./models
COPY templates ./templates
COPY retry ./retry
COPY insightful_ai_api.py operation_result.py ./

# Build wheel with all dependencies
RUN pip install --upgrade pip wheel && \
    pip wheel --no-cache-dir --wheel-dir /wheels .


FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 insightful

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install runtime dependencies and InsightfulAI
RUN pip install --no-cache-dir --find-links /wheels InsightfulAI && \
    rm -rf /wheels

# Set ownership
RUN chown -R insightful:insightful /app

# Switch to non-root user
USER insightful

# InsightfulAI ML framework
ENTRYPOINT ["python", "-c", "print('InsightfulAI v0.3.0a1 ready')"]
CMD []
