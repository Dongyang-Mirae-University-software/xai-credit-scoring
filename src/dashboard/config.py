"""대시보드 설정값. 환경 변수로 덮어쓸 수 있다."""
import os

# 미션 필수 면책 조항 (원본 과제 문구 그대로, API 응답의 disclaimer 필드와 같은 문장)
DISCLAIMER = "본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다."

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
HEALTH_TIMEOUT = float(os.getenv("HEALTH_TIMEOUT", "2"))
HEALTH_CACHE_TTL = int(os.getenv("HEALTH_CACHE_TTL", "10"))
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "5"))
# API 호출 로그: "" 끔 / "1" 한 줄 요약 / "body" 요청·응답 JSON 까지 (api_client.request)
API_LOG = os.getenv("API_LOG", "").strip().lower()
API_LOG = "" if API_LOG in ("", "0", "false", "off") else ("body" if API_LOG == "body" else "1")
API_LOG_BODY_MAX = int(os.getenv("API_LOG_BODY_MAX", "600"))

# ---- 고객 심사 화면 표시용 값 ----
# 점수·등급 기준 (README '점수와 등급 기준'): credit_score = round((1 - 부도 확률) × 1000)
# 등급 구간 (등급, 구간 상한 점수)은 API(src/api/model.py GRADE_SCALE)와 같다:
# E 300 미만 · D 300~499 · C 500~649 · B 650~799 · A 800 이상. API 가 구간을 내려주지 않아 여기에 둔다
# 결과 화면 등급 막대의 칸 폭(30:20:15:15:20)은 이 구간 길이를 그대로 쓴다
GRADE_SCALE = [("E", 300), ("D", 500), ("C", 650), ("B", 800), ("A", 1000)]
# 승인 기준(막대의 점선)은 여기 두지 않는다: /predict 응답(없으면 /health)의 approval_threshold 로 계산한다

# ---- XAI 분석 화면 표시용 값 ----
# 거절 사유 카드의 위험 표시: 부도 쪽 SHAP 값(log-odds)이 이 값 이상이면 고위험 / 중위험, 그 아래는 저위험
XAI_RISK_HIGH = float(os.getenv("XAI_RISK_HIGH", "0.7"))
XAI_RISK_MID = float(os.getenv("XAI_RISK_MID", "0.4"))
# Waterfall에 따로 보여줄 변수 수 (나머지는 '기타 N개 변수'로 합침)
WATERFALL_MAX_DISPLAY = int(os.getenv("WATERFALL_MAX_DISPLAY", "8"))
# 가법성 검증 허용 오차: |sigmoid(base + Σφ) - /predict 부도 확률|
ADDITIVITY_TOLERANCE = float(os.getenv("ADDITIVITY_TOLERANCE", "0.005"))
