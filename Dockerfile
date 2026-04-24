# 1. NVIDIA CUDA와 PyTorch가 설치된 이미지를 베이스로 사용
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# 환경 변수 설정 (파이썬 출력 버퍼링 제거 및 인터랙티브 설치 방지)
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 2. 필수 시스템 패키지 설치
# - portaudio19-dev: PyAudio 설치 및 마이크 입력에 필수
# - libsndfile1, ffmpeg: librosa 및 오디오 처리에 필수
# - python3-dev, gcc: 일부 라이브러리 컴파일 설치에 필요
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    python3-dev \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 파이썬 라이브러리 설치
# (변경 사항이 없을 시 캐시를 활용하기 위해 복사를 먼저 진행)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. 컨테이너 실행 시 기본 실행 파일
# (현재는 무한 대기 상태가 필요할 수 있으므로, 나중에 main.py가 완성되면 그대로 사용하세요)
CMD ["python", "src/main.py"]