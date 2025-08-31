# セキュリティ: 画像は保存しない前提（後述の compose で read-only + tmpfs を適用）
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCV_MODEL_PATH=/opt/opencv_models

# 必須ランタイム
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-jpn tesseract-ocr-eng \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# OpenCV Haar を同梱
RUN mkdir -p /opt/opencv_models && \
    wget -qO /opt/opencv_models/haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml && \
    wget -qO /opt/opencv_models/haarcascade_frontalface_alt.xml     https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml && \
    wget -qO /opt/opencv_models/haarcascade_profileface.xml         https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_profileface.xml

WORKDIR /app

# Python 依存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体
COPY app/ app/

# 非rootユーザ
RUN adduser --disabled-password --gecos '' --home /nonexistent --no-create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# HEALTHCHECK（requests不要）
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8000/health >/dev/null 2>&1 || exit 1

# アクセスログ抑止（PII低減）
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000", "--workers=1", "--no-access-log"]
