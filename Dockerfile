# Dockerfile

# Stage 1: Build Stage (using a robust Python base image)
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy only the necessary files for dependency installation
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Image
FROM python:3.11-slim

# Set environment variables
ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH=${PYTHONPATH}:${APP_HOME}

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code, configuration, and MLflow data
# NOTE: The MLflow tracking DB must be copied if it's local (sqlite:///mlruns.db)
COPY . $APP_HOME

# Expose the port the application runs on
EXPOSE 8000

# Command to run the application using uvicorn
# Note: Host 0.0.0.0 is essential for container access
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]