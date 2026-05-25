FROM python:3.11-slim

WORKDIR /app

# Install system libraries needed by OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required folders
RUN apt-get update && apt-get install -y curl
RUN mkdir -p database webapp/static/uploads models


EXPOSE 5000

CMD ["gunicorn", "webapp.app:create_app()", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--timeout", "120"]