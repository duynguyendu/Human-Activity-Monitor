# Using ubuntu 22.04 base image
FROM ubuntu:22.04

# Set up system
RUN apt update && \
    # Install opencv dependencies
    apt install ffmpeg libsm6 libxext6 -y && \
    # Install python and pip
    apt install python3.10 python3-pip -y && \
    # Clean up
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create working dir
WORKDIR /app
COPY . /app

# Install required packages
RUN pip install --no-cache --upgrade pip setuptools wheel && \
    pip install --no-cache -r requirements.txt
