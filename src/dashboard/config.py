"""대시보드 설정값. 환경 변수로 덮어쓸 수 있다."""
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
HEALTH_TIMEOUT = float(os.getenv("HEALTH_TIMEOUT", "2"))
HEALTH_CACHE_TTL = int(os.getenv("HEALTH_CACHE_TTL", "10"))
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "5"))

# true면 API 대신 mock/*.json 응답을 사용한다 (백엔드 완성 전 화면 개발용)
USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"
