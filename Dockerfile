# ---------- Runtime slim ----------
FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# (tùy chọn) cài gói hệ thống cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài deps trước để tối ưu cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Non-root user (tùy chọn, an toàn hơn)
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# HEALTHCHECK nội bộ container (dùng curl từ busybox python? -> dùng uvicorn + httpx là overkill;
# ta để healthcheck ở docker-compose để curl từ host container)
# ENTRYPOINT/CMD
CMD ["uvicorn", "fp_rec_api_new:app", "--host", "0.0.0.0", "--port", "8000"]
