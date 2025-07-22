FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# RunPod requires this handler
CMD ["python", "handler.py"]
