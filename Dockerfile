# Start from an official Python 3.9 image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install FFmpeg
# This runs the system commands we need
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 2. Copy and install Python requirements
# This copies *only* the requirements file first, to cache this layer
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 3. Copy the rest of your app code
COPY . .

# 4. Set the command to run your app
# This replaces Render's "Start Command"
CMD ["streamlit", "run", "app.py"]
