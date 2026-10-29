"""대시보드 설정값. 환경 변수로 덮어쓸 수 있다."""
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
HEALTH_TIMEOUT = float(os.getenv("HEALTH_TIMEOUT", "2"))
HEALTH_CACHE_TTL = int(os.getenv("HEALTH_CACHE_TTL", "10"))
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "5"))

# true면 API 대신 mock/*.json 응답을 사용한다 (백엔드 완성 전 화면 개발용)
USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"

# ---- 고객 심사 화면 표시용 값 (미션 문서에 정의 없음 → 백엔드 확정 시 동기화) ----
# 승인 임계값: 부도 확률이 이 값 미만이면 승인
APPROVAL_THRESHOLD = float(os.getenv("APPROVAL_THRESHOLD", "0.5"))
# 신용점수 막대 아래 표시하는 등급 구간 (등급, 구간 상한 점수)
GRADE_SCALE = [("E", 300), ("D", 500), ("C", 650), ("B", 800), ("A", 1000)]

# ---- XAI 분석 화면 표시용 값 ----
# 거절 사유 카드의 위험 표시: 부도 쪽 SHAP 값(log-odds)이 이 값 이상이면 고위험 / 중위험, 그 아래는 저위험
XAI_RISK_HIGH = float(os.getenv("XAI_RISK_HIGH", "0.7"))
XAI_RISK_MID = float(os.getenv("XAI_RISK_MID", "0.4"))
# Waterfall에 따로 보여줄 변수 수 (나머지는 '기타 N개 변수'로 합침)
WATERFALL_MAX_DISPLAY = int(os.getenv("WATERFALL_MAX_DISPLAY", "8"))
# 가법성 검증 허용 오차: |sigmoid(base + Σφ) − /predict 부도 확률|
ADDITIVITY_TOLERANCE = float(os.getenv("ADDITIVITY_TOLERANCE", "0.005"))
