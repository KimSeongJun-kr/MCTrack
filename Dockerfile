# 1. 베이스 이미지 설정
# 공식 Python 3.8 슬림 버전을 사용하여 이미지 크기를 최적화합니다.
FROM python:3.8

# 2. 작업 디렉토리 설정
# 컨테이너 내에서 명령이 실행될 기본 디렉토리를 설정합니다.
WORKDIR /3dmot_ws/MCTrack

RUN apt-get update && \
    apt-get install -y build-essential libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# RUN pip install --upgrade pip && \
#     pip install numpy==1.22.0 && \
#     pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade pip && \
    pip install numpy==1.22.0 Cython && \
    pip install --no-cache-dir --no-binary :all: --no-build-isolation lap && \
    pip install --no-cache-dir -r requirements.txt