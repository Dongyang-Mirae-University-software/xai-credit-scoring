"""
API 서버 HTTP 클라이언트.
MSA 제약: 대시보드는 모델 파일을 직접 로드하지 않고 이 모듈을 통해서만 데이터를 받는다.

리포트형 API(/metrics, /explain/summary, /fairness, /fairness/mitigation, /anova)는 파이프라인 산출물(outputs/)이
없으면 서버가 예시(src/api/examples/) 값을 대신 돌려준다. 예시 값을 실제 결과처럼 보여 주지 않도록,
/health 의 reports 에서 'example' / 'missing' 이면 호출하지 않고 None 을 돌려준다 → 화면은 '데이터 없음' 안내.
"""
import time

import requests

import config

NO_DATA = {"example", "missing"}
_health_cache = {"at": 0.0, "data": None}


def _get(path):
    r = requests.get(f"{config.API_BASE_URL}{path}", timeout=config.API_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _post(path, payload):
    r = requests.post(f"{config.API_BASE_URL}{path}", json=payload, timeout=config.API_TIMEOUT)
    r.raise_for_status()
    return r.json()


def health():
    # 한 화면에서 리포트마다 확인하므로 잠깐 재사용한다 (사이드바와 같은 TTL)
    if _health_cache["data"] is None or time.monotonic() - _health_cache["at"] > config.HEALTH_CACHE_TTL:
        _health_cache["data"] = _get("/health")
        _health_cache["at"] = time.monotonic()
    return _health_cache["data"]


def report_status(name):
    """/health 의 reports[name]: 산출물 파일 경로, 'example' 또는 'missing'. 서버가 알려 주지 않으면 None."""
    return health().get("reports", {}).get(name)


def _report(name, path):
    """파이프라인 산출물이 있을 때만 받아 온다. 없으면 None."""
    if report_status(name) in NO_DATA:
        return None
    return _get(path)


def predict(customer):
    return _post("/predict", customer)


def explain(customer):
    return _post("/explain", customer)


def shap_summary():
    """
    전역 중요도 (GET /explain/summary). 산출물이 없으면 None.
    형식: {"model": str, "n_samples": int, "shap_space": str, "features": [{feature, mean_abs_shap, group}]}
    """
    return _report("shap_summary", "/explain/summary")


def fairness():
    return _report("fairness", "/fairness")


def fairness_mitigation():
    return _report("fairness_mitigation", "/fairness/mitigation")


def metrics():
    return _report("metrics", "/metrics")


def model_comparison():
    """
    후보 모델 격자 비교 (GET /metrics/comparison). models/model_metadata.json 에서 오므로 항상 실제 값.
    '± 표준편차' 표기용 cv_runs[*].cv_test_std 는 백엔드에 아직 없어, 없으면 평균만 표시한다.
    """
    return _get("/metrics/comparison")


def anova():
    return _report("anova", "/anova")
