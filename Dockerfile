# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=10000 \
    ENVIRONMENT=production \
    MODEL_PATH=/app/models/best.pt \
    UPLOAD_DIR=/app/uploads \
    TEMP_DIR=/app/temp

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/uploads /app/temp /app/models

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Copy model file (if exists)
COPY models/best.pt /app/models/best.pt

# Expose the port
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/ || exit 1

# Command to run the application
# CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port $PORT"

