



# 1) 베이스 이미지 선택 (파이썬 3.9 사용)
FROM python:3.11-slim

# 2) 작업 디렉토리 설정
WORKDIR /main

# [중요] 오디오 처리(Librosa 등)를 위한 시스템 라이브러리 설치
# (이게 없으면 오디오 파일 읽을 때 에러납니다)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3) 필요한 파일들 복사
# 먼저 requirements.txt만 복사해서 라이브러리 설치 (캐시 활용 속도 향상)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) 나머지 소스 코드 및 모델 파일 복사
# (현재 폴더의 모든 파일을 도커 내부 /app으로 복사)
COPY . .

# 5) 컨테이너에서 열어둘 포트 (FastAPI 기본 8000)
EXPOSE 8000

# 6) 컨테이너 실행 시 돌아갈 커맨드
# (main.py 안의 app 객체를 실행)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]