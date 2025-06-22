# Use an official Python runtime as a parent image
FROM python:3.12.11

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained YOLO model (if it's in the same directory as Dockerfile)
# Adjust this path if your model is elsewhere relative to the Dockerfile
COPY defect_detector_model.pt .

# Copy the Python script
COPY main.py .

# Create directories for input and output images
RUN mkdir -p unprocessed_images
RUN mkdir -p processed_images

# Command to run the application (assuming your script is named main.py)
# This will be the default command executed when the container starts
CMD ["python", "main.py"]
