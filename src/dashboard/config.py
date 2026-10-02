"""대시보드 설정값. 환경 변수로 덮어쓸 수 있다."""
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
HEALTH_TIMEOUT = float(os.getenv("HEALTH_TIMEOUT", "2"))
HEALTH_CACHE_TTL = int(os.getenv("HEALTH_CACHE_TTL", "10"))
