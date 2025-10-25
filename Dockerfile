FROM python:3.9-slim

WORKDIR /app

# Install build-essential
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install with pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all app code
COPY . .

# Create models, uploads, data directories
RUN mkdir -p models uploads data

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port 5000
EXPOSE 5000

# CMD with gunicorn bind 0.0.0.0:5000 workers 4 timeout 120
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
