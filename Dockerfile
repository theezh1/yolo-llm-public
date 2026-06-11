# Single image used for both services (bot + vision-server).
# Based on the official PyTorch CUDA runtime — same family as RunPod's PyTorch preset.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libs needed by Pillow / ultralytics (OpenCV) and a font for annotations.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install both dependency sets. torch is already in the base image; pip skips it.
COPY requirements.txt /tmp/bot-req.txt
COPY vision_server/requirements.txt /tmp/vision-req.txt
RUN pip install -r /tmp/bot-req.txt -r /tmp/vision-req.txt

COPY . /app

# Default command runs the bot; docker-compose overrides for the vision-server.
CMD ["python", "bot.py"]
