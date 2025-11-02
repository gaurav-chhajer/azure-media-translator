# Start from an official Python 3.9 image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install FFmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 2. Copy and install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 3. Copy the rest of your app code
COPY . .

# 4. (NEW) Copy the start script and make it executable
COPY start.sh .
RUN chmod +x start.sh

# 5. (NEW) Set the command to run the start script
CMD ["./start.sh"]
