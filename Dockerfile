# ── Housing Price Prediction API ─────────────────────────────────────────────
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies (pre-built wheels — no compiler needed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and data
COPY app/ ./app/
COPY data/ ./data/

# Create models directory
RUN mkdir -p models

# Train the model at build time so the container is fully self-contained
RUN python -m app.train

# Expose the API port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
