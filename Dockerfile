# 1. 가벼운 파이썬 공식 이미지를 사용 (NVIDIA CUDA 제외)
FROM python:3.10-slim

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 2. 필수 시스템 패키지 설치 (오디오 처리를 위한 최소 패키지)
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. 파이썬 라이브러리 설치
COPY requirements.txt .
# CPU 버전 torch를 설치하도록 유도 (용량이 훨씬 작음)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 복사
COPY . .

CMD ["python", "src/main.py"]