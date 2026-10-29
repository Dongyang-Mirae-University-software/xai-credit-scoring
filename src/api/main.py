"""
FastAPI 신용평가 API.

대시보드(Streamlit) 5개 화면이 쓰는 엔드포인트를 화면 순서대로 태그로 묶었다.
- /predict, /explain 은 models/ 의 최종 모델로 바로 계산한다.
- 나머지는 파이프라인(main_step3~5) 산출물 JSON 을 그대로 내보내며, 파일이 없으면 내장 예시를 돌려준다 (/health 의 reports 참고).

실행: uvicorn src.api.main:app --reload  →  http://localhost:8000/docs
"""
from fastapi import FastAPI

from src.api.routers import anova, explain, fairness, health, metrics, predict

DESCRIPTION = """
XAI 기반 대안신용평가 시스템 API. 대시보드는 모델 파일을 직접 열지 않고 이 API 로만 통신한다.

| 화면 | 엔드포인트 | 데이터 출처 |
|---|---|---|
| 사이드바 | `GET /health` | 모델 파일 존재 여부, 리포트 출처 |
| 01 고객 심사 | `POST /predict` | 최종 모델 즉시 계산 |
| 02 XAI 분석 | `POST /explain`, `GET /explain/summary` | SHAP 즉시 계산 / `outputs/evaluation/shap_summary.json` |
| 03 공정성 | `GET /fairness`, `GET /fairness/mitigation` | `outputs/fairness/*.json` |
| 04 모델 성능 | `GET /metrics`, `GET /metrics/comparison` | `outputs/evaluation/metrics_*.json`, `models/model_metadata.json` |
| 05 통계 검증 | `GET /anova` | `outputs/statistics/anova.json` |

> 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.
"""

TAGS = [
    {'name': '시스템', 'description': '상태 확인'},
    {'name': '01 고객 심사', 'description': '고객 입력 → 신용점수·등급·승인 여부'},
    {'name': '02 XAI 분석', 'description': 'SHAP 전역 중요도, 개별 고객 Waterfall, 거절 사유 Top 5'},
    {'name': '03 공정성', 'description': 'DI Ratio · Equalized Odds, Bias Mitigation 전/후'},
    {'name': '04 모델 성능', 'description': 'AUC·KS·PSI, 혼동행렬, ROC/PR 곡선, 후보 모델 비교'},
    {'name': '05 통계 검증', 'description': 'One-way ANOVA · Tukey HSD'},
]

app = FastAPI(title='XAI 기반 대안신용평가 API', version='1.0.0', description=DESCRIPTION, openapi_tags=TAGS)
for module in (health, predict, explain, fairness, metrics, anova):
    app.include_router(module.router)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
