"""공정성 기준 판정과 지표 숫자 표시. 화면 코드 없이 테스트할 수 있게 views/fairness.py 에서 분리했다.

기준은 src/fairness/metrics.py 와 같다.
- 승인율 비율(DI): di_min 이상 di_max 이하면 통과 (di_max 가 없으면 아래쪽만)
- 오류율 차이(Equalized Odds TPR · FPR): eo_max 이하면 통과
"""


def di_ok(value, criteria):
    if value is None:
        return None
    return value >= criteria["di_min"] and (criteria.get("di_max") is None or value <= criteria["di_max"])


def eo_ok(value, criteria):
    return None if value is None else value <= criteria["eo_max"]


def fmt_value(value, ok, criteria):
    """소수 둘째 자리. 단 미달인데 반올림하면 기준 경계값과 같아 보이면(예: 1.2535 → 1.25) 셋째 자리까지."""
    if value is None:
        return "-"
    bounds = {round(criteria[k], 2) for k in ("di_min", "di_max", "eo_max") if criteria.get(k) is not None}
    return f"{value:.3f}" if ok is False and round(value, 2) in bounds else f"{value:.2f}"
