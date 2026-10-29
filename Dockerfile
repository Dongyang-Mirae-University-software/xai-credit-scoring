FROM python:3.10-slim

WORKDIR /app

# lightgbm 실행에 OpenMP 런타임이 필요하다 (slim 이미지에는 없음)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
