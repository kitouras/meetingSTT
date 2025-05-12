FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

COPY diarization_service/requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir -r requirements.txt

COPY diarization_service ./diarization_service
COPY settings.json .

RUN git clone https://github.com/salute-developers/GigaAM.git && \
    pip3 install --no-cache-dir -e ./GigaAM

RUN python3 ./diarization_service/pre_cache_gigaam.py
RUN python3 ./diarization_service/pre_cache_pyannote.py

RUN mkdir -p diarization_uploads

EXPOSE 5002

CMD ["python3", "-m", "diarization_service.api"]