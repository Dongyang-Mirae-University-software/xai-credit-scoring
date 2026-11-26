"""
API 서버 HTTP 클라이언트.
MSA 제약: 대시보드는 모델 파일을 직접 로드하지 않고 이 모듈을 통해서만 데이터를 받는다.

리포트형 API(/metrics, /explain/summary, /fairness, /fairness/mitigation, /anova)는 파이프라인 산출물(outputs/)이
없으면 서버가 예시(src/api/examples/) 값을 대신 돌려준다. 예시 값을 실제 결과처럼 보여 주지 않도록,
/health 의 reports 에서 'example' / 'missing' 이면 호출하지 않고 None 을 돌려준다 → 화면은 '데이터 없음' 안내.
"""
import json
import logging
import sys
import time

import requests

import config

NO_DATA = {"example", "missing"}
_health_cache = {"at": 0.0, "data": None}

# API 호출 로그 (Streamlit 은 서버에서 API 를 부르므로 브라우저 네트워크 탭에 안 보인다 → 터미널에 찍는다)
#   API_LOG=1    → POST /predict 200 (85ms)
#   API_LOG=body → 위 줄 + 요청·응답 JSON. 목록 응답은 한 건씩 한 줄로
#   API_LOG_BODY_MAX → 한 줄 최대 글자 수 (기본 600, 0 이면 자르지 않는다)
_log = logging.getLogger("dashboard.api")
if config.API_LOG and not _log.handlers:  # Streamlit 이 모듈을 다시 읽어도 핸들러는 하나만
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s [API] %(message)s", "%H:%M:%S"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)
    _log.propagate = False


SECRET_KEYS = {"rrn"}  # 로그에 그대로 남기지 않는 요청 칸 (주민등록번호)


def _mask(data):
    if isinstance(data, dict):
        return {k: ("******" if k in SECRET_KEYS else _mask(v)) for k, v in data.items()}
    return data


def _short(data):
    data = _mask(data)
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    limit = config.API_LOG_BODY_MAX
    return text if limit <= 0 or len(text) <= limit else text[:limit] + f"… ({len(text)}자)"


def _log_body(label, data):
    """목록이면 '응답 목록 N건' 아래에 한 건씩, 아니면 한 줄."""
    if isinstance(data, list):
        _log.info("  %s 목록 %d건", label, len(data))
        for i, item in enumerate(data, 1):
            _log.info("    %d. %s", i, _short(item))
    else:
        _log.info("  %s %s", label, _short(data))


def request(method, path, payload=None, timeout=None):
    """API 호출 하나. 실패하면 requests 예외를 그대로 올린다. API_LOG 가 켜져 있으면 결과를 터미널에 남긴다."""
    started = time.perf_counter()
    try:
        r = requests.request(method, f"{config.API_BASE_URL}{path}", json=payload,
                             timeout=timeout or config.API_TIMEOUT)
    except requests.RequestException as e:
        if config.API_LOG:
            _log.info("%s %s 연결 실패 (%dms) %s", method, path, (time.perf_counter() - started) * 1000,
                      type(e).__name__)
        raise
    if config.API_LOG:
        _log.info("%s %s %d (%dms)", method, path, r.status_code, (time.perf_counter() - started) * 1000)
        if config.API_LOG == "body":
            if payload is not None:
                _log_body("요청", payload)
            try:
                _log_body("응답", r.json())
            except ValueError:
                _log_body("응답", r.text)
    r.raise_for_status()
    return r.json()


def _get(path):
    return request("GET", path)


def _post(path, payload):
    return request("POST", path, payload)


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


# ---------------------------------------------------------------------------
# 심사 이력 (심사 이력이 새로고침해도 남도록). 서버가 예전 버전이라 없으면 404 → False / [] 로 대비
# ---------------------------------------------------------------------------
def _missing(exc):
    return isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code in (404, 405)


def save_screening(record):
    """POST /screenings: 심사 결과 저장 ({input, result}). API 가 아직 없으면 False."""
    try:
        _post("/screenings", record)
        return True
    except requests.HTTPError as e:
        if _missing(e):
            return False
        raise


def screenings(limit=20):
    """GET /screenings?limit=: 저장된 최근 심사 (최신순, 고객당 1건). API 가 아직 없으면 [].
    항목: {screening_id, customer_id, approval_status, risk_grade, credit_score, screened_at(ISO)}"""
    try:
        return _get(f"/screenings?limit={limit}")
    except requests.HTTPError as e:
        if _missing(e):
            return []
        raise


# ---------------------------------------------------------------------------
# 보너스 과제 결과 (#140). API 가 아직 없으면(404) NOT_READY 를 돌려준다 → 화면은 'API 준비 중' 안내
#   산출물이 없으면 None (/health 의 reports 에 'missing' 으로 오면). 다른 오류는 그대로 올린다
# ---------------------------------------------------------------------------
NOT_READY = "not_ready"


def _bonus(name, path):
    try:
        return _report(name, path)
    except requests.HTTPError as e:
        if _missing(e):
            return NOT_READY
        raise


def counterfactual(customer):
    """POST /counterfactual (보너스 2, #147): 거절 고객이 승인되려면 바꿔야 할 정보.
    형식: {approval_status, default_probability, approval_threshold, already_approved,
           combined: {success, changes[], new_probability, new_credit_score} | None,
           single_changes[], immutable_features[]}
    changes[]: {feature, feature_name_kr, current_value, suggested_value, unit, explanation, new_probability}"""
    try:
        return request("POST", "/counterfactual", customer, timeout=config.COUNTERFACTUAL_TIMEOUT)
    except requests.HTTPError as e:
        if _missing(e):
            return NOT_READY
        raise


def drift():
    """GET /drift (보너스 4, #149 · API #139): outputs/monitoring/drift_report.json 에서 구간별 상세(tables)를 뺀 것.
    형식: {thresholds: {stable, caution}, scenarios: {키: {name, score_psi, score_grade, csi: {칸: {index, grade}},
           flagged[], top_bins: {칸: {label, expected, actual}}, auc, approval_rate, thin_filer_share}}}"""
    return _bonus("drift", "/drift")


def two_way_anova():
    """GET /anova/two-way (보너스 3, #148): outputs/statistics/two_way_anova.json 에서 records 를 뺀 것.
    형식: {design, anova: {table[], n_per_cell, cell_means, cell_std}, paired_checks: {simple_effects, interaction_did},
           interpretation[]}"""
    return _bonus("two_way_anova", "/anova/two-way")


def ab_test():
    """GET /ab-test (보너스 5, #150): outputs/evaluation/ab_test/ab_test.json 그대로.
    형식: {models: {a, b}, test_auc: {a, b}, policies: {a, threshold_a, policies: {same_threshold, same_approval_rate,
           same_bad_rate}}, ab_single, ab_repeated, required_sample_size}"""
    return _bonus("ab_test", "/ab-test")


def lookup_customer(name, rrn):
    """POST /customers/lookup {name, rrn}: 은행이 가진 그 고객의 신용정보 · 대안정보 (본인 확인 → 정보 조회).
    형식: {customer_id, name, gender, + POST /predict 입력 15칸} (단위는 /predict 와 같다)
    본인 정보가 없으면 None, API 가 아직 없으면 NOT_READY.
    주민등록번호는 URL 이 아니라 본문으로 보낸다 (URL 은 서버 접근 로그에 남는다)."""
    try:
        return _post("/customers/lookup", {"name": name, "rrn": rrn})
    except requests.HTTPError as e:
        response = e.response
        if response is None:
            raise
        if response.status_code == 405:  # GET /customers/{id} 만 있을 때
            return NOT_READY
        if response.status_code == 404:
            try:
                route_missing = response.json().get("detail") == "Not Found"  # FastAPI 기본 404 = 경로 없음
            except ValueError:
                route_missing = True
            return NOT_READY if route_missing else None
        raise
