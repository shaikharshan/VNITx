FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV SPACE_MODE=1

EXPOSE 7860

# Image_prompt_detection/Dockerfile (CMD line)
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
