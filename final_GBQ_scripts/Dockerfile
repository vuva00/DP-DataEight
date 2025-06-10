"""Dockerfile for Google BigQuery scripts
This Dockerfile sets up a lightweight Python environment to run Google BigQuery scripts."""

# Use a lightweight base Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy scripts and requirements
COPY requirements.txt .
COPY model_run_weekly.py .
COPY model_train_run_monthly.py .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Default command can be overridden in the job definition
ENTRYPOINT ["python"]