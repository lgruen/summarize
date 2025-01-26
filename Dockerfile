FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .

# Runtime configuration
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=2

# Run as non-root user
RUN useradd -m appuser
USER appuser

CMD ["python", "main.py"]
