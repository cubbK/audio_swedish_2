# Use the same base image as Vertex AI training
FROM europe-docker.pkg.dev/vertex-ai/training/tf-gpu.2-17.py310:latest

# Install ffmpeg and other audio processing tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy your application code


# Install Python dependencies if you have any
# COPY requirements.txt .
# RUN pip install -r requirements.txt
RUN pip install openai-whisper pandas datasets[audio] huggingface_hub
RUN pip install torchcodec --index-url=https://download.pytorch.org/whl/cu128
RUN pip install scipy librosa
RUN pip install "datasets>=2.14.0" "transformers>=4.30.0" "accelerate>=0.26.0" "torch>=2.0.0" "google-cloud-storage>=2.0.0" "tf-keras"


COPY . /app
WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Set the entrypoint
ENTRYPOINT ["python", "-u", "job.py"]