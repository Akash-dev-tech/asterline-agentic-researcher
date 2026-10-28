# ==============================================================================
# Asterline Labs - Production Dockerfile
# Optimized for lightweight CPU inference & fast cloud container startup
# ==============================================================================

FROM python:3.10-slim

# Set environment variables for Python runtime and Docker optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory inside container
WORKDIR /app

# Install minimal OS dependencies required for native builds & health probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and packaging tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install CPU-only PyTorch first to prevent downloading heavy CUDA binaries (>2GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and assets
COPY . .

# Ensure runtime directories exist with proper write permissions
RUN mkdir -p /app/data /app/data/chroma_store

# Expose default application port
EXPOSE 8000

# Execute Uvicorn server dynamically binding to the host or platform $PORT
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
