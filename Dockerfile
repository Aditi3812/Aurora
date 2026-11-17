FROM python:3.11-slim

WORKDIR /app

# System deps for ML packages (scikit-learn, pandas, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libopenblas-dev && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and dataset
COPY . .

# Build artifacts at IMAGE BUILD TIME (cached in layer)
RUN python model.py

# Streamlit runs on port 7860 (Render expects this for Docker)
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]