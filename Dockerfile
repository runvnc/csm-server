FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server.py .

# CSM model path (mount or copy model files)
ENV CSM_MODEL_PATH=/models/csm
ENV CSM_DEVICE=cuda
ENV CSM_PORT=8765
ENV CSM_CODEBOOKS=32
ENV CSM_PATH=/app/csm-streaming

# Expose port
EXPOSE 8765

# Run server
CMD ["python", "server.py"]
