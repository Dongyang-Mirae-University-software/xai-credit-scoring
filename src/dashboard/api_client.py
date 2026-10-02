"""
API 서버 HTTP 클라이언트.
MSA 제약: 대시보드는 모델 파일을 직접 로드하지 않고 이 모듈을 통해서만 데이터를 받는다.

config.USE_MOCK이 true면 mock/<페이지>/ 폴더의 JSON을 반환한다.
mock JSON의 키 이름은 미션 문서의 API 응답 형식과 동일하므로,
백엔드가 완성되면 USE_MOCK=false로만 바꾸면 된다.
"""
import copy
import json
from functools import lru_cache
from pathlib import Path

import requests

import config

MOCK_DIR = Path(__file__).parent / "mock"


@lru_cache(maxsize=None)
def _load_mock(page, name):
    return json.loads((MOCK_DIR / page / f"{name}.json").read_text(encoding="utf-8"))


def _mock(page, name):
    return copy.deepcopy(_load_mock(page, name))


def _is_rejected(customer):
    """mock 전용: 입력값에 따라 승인/거절 예시 중 하나를 고른다."""
    return (
        customer.get("NumberOfTimes90DaysLate", 0) > 0
        or customer.get("telecom_payment_rate", 1.0) < 0.85
    )


def _mock_for_customer(page, prefix, customer):
    data = _mock(page, f"{prefix}_rejected" if _is_rejected(customer) else f"{prefix}_approved")
    data["customer_id"] = customer.get("customer_id", data["customer_id"])
    return data


def _get(path):
    r = requests.get(f"{config.API_BASE_URL}{path}", timeout=config.API_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _post(path, payload):
    r = requests.post(f"{config.API_BASE_URL}{path}", json=payload, timeout=config.API_TIMEOUT)
    r.raise_for_status()
    return r.json()


def health():
    return _mock("sidebar", "health") if config.USE_MOCK else _get("/health")


def predict(customer):
    return _mock_for_customer("screening", "predict", customer) if config.USE_MOCK else _post("/predict", customer)


def explain(customer):
    return _mock_for_customer("xai", "explain", customer) if config.USE_MOCK else _post("/explain", customer)


def fairness():
    return _mock("fairness", "fairness") if config.USE_MOCK else _get("/fairness")


def fairness_mitigation():
    return _mock("fairness", "fairness_mitigation") if config.USE_MOCK else _get("/fairness/mitigation")


def metrics():
    return _mock("performance", "metrics") if config.USE_MOCK else _get("/metrics")


def anova():
    return _mock("statistical_test", "anova") if config.USE_MOCK else _get("/anova")
