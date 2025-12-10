FROM python:3.11-slim

WORKDIR /app

# 필수 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# Streamlit 포트 개방
EXPOSE 8501

# 실행 명령어
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]