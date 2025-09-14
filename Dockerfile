# Multi-stage Dockerfile for QSFL-CAAD
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Development stage
FROM base as development

# Install development dependencies
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt \
    && pip install --no-cache-dir -r requirements-test.txt \
    && pip install --no-cache-dir -r requirements-ui.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p logs data/exports ui/static/images \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 5000 8000

# Default command
CMD ["python", "working_dashboard.py"]

# Production stage
FROM base as production

# Install only production dependencies
COPY requirements.txt requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-ui.txt

# Copy source code
COPY qsfl_caad/ ./qsfl_caad/
COPY ui/ ./ui/
COPY scripts/ ./scripts/
COPY setup.py pyproject.toml ./

# Install package
RUN pip install .

# Create necessary directories
RUN mkdir -p logs data/exports ui/static/images \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Production command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--worker-class", "eventlet", "ui.working_dashboard:app"]

# Testing stage
FROM development as testing

# Copy test files
COPY tests/ ./tests/

# Run tests
RUN python -m pytest tests/ --tb=short

# Jupyter stage
FROM base as jupyter

# Install Jupyter and additional packages
RUN pip install --no-cache-dir \
    jupyterlab \
    notebook \
    ipywidgets \
    matplotlib \
    seaborn \
    plotly \
    pandas \
    numpy \
    scikit-learn

# Copy requirements and install
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create jupyter user
RUN useradd -m -s /bin/bash jupyter

# Set up Jupyter
USER jupyter
WORKDIR /home/jupyter

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]