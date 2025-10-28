FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

COPY diarization_service/requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY settings.json .
COPY diarization_service/pre_cache_pyannote.py .
COPY diarization_service/pre_cache_whisper.py .

RUN python3 ./pre_cache_pyannote.py
RUN python3 ./pre_cache_whisper.py

COPY diarization_service ./diarization_service

RUN mkdir -p diarization_uploads

EXPOSE 5002

CMD ["python3", "-m", "diarization_service.api"]