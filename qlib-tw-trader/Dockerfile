FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for LightGBM and qlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ src/
COPY pyproject.toml .

# Create data directories (will be mounted as volumes)
RUN mkdir -p data data/models data/qlib

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Taipei

EXPOSE 8000

CMD ["uvicorn", "src.interfaces.app:app", "--host", "0.0.0.0", "--port", "8000"]
