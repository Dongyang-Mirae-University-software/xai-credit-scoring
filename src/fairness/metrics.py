"""
공정성 지표: 집단별 승인율, Disparate Impact Ratio, Equalized Odds.

원본 과제 기준
- 「Disparate Impact Ratio(DI Ratio)를 계산하는 함수가 존재한다.」
- 「DI Ratio 계산 결과가 0.8 미만 또는 1.25 초과일 때 WARNING 로그가 출력된다.」
- 「Equalized Odds를 계산하는 함수가 존재한다 (TPR 차이, FPR 차이).」
- 「보호 속성으로 성별(male/female)과 연령대(20-34/35-54/55+)를 사용한다.」
- DI Ratio 정의: 「보호 집단의 승인율을 비보호 집단의 승인율으로 나눈 비율. 0.8 미만 시 차별적 영향이 있는 것으로 판단」

용어
- 승인: 부도 확률이 기준값(APPROVAL_THRESHOLD, 기본 0.5)보다 낮은 사람
- Equalized Odds 는 '승인'을 좋은 결과로 본다.
  TPR = 실제 정상 상환자 중 승인된 비율, FPR = 실제 연체자 중 승인된 비율.
  두 집단의 TPR 차이와 FPR 차이가 모두 0.10 이하이면 기준 충족.

기준값은 환경 변수로 바꿀 수 있다: FAIRNESS_DI_MIN, FAIRNESS_DI_MAX, FAIRNESS_EO_MAX,
FAIRNESS_PROTECTED_GENDER, FAIRNESS_PROTECTED_AGE.
"""

import logging
import os

logger = logging.getLogger(__name__)

DI_MIN = float(os.getenv('FAIRNESS_DI_MIN', '0.8'))
DI_MAX = float(os.getenv('FAIRNESS_DI_MAX', '1.25'))
EO_MAX = float(os.getenv('FAIRNESS_EO_MAX', '0.10'))

GENDERS = ('male', 'female')
AGE_GROUPS = ('20-34', '35-54', '55+')

# 속성별 보호 집단 (나머지가 비보호 집단)
PROTECTED = {
    'gender': os.getenv('FAIRNESS_PROTECTED_GENDER', 'female'),
    'age_group': os.getenv('FAIRNESS_PROTECTED_AGE', '55+'),
}

# docs/column_spec.md: Gender 0 = 여성, 1 = 남성
GENDER_CODES = {0: 'female', 1: 'male', '0': 'female', '1': 'male',
                'f': 'female', 'm': 'male', 'female': 'female', 'male': 'male'}


def _list(values):
    return values.tolist() if hasattr(values, 'tolist') else list(values)


def default_threshold():
    return float(os.getenv('APPROVAL_THRESHOLD', '0.5'))


def normalize_gender(values):
    """0/1, 'F'/'M', 'female'/'male' 을 'female'/'male' 로 맞춘다."""
    out = []
    for v in _list(values):
        key = v.strip().lower() if isinstance(v, str) else (int(v) if v == v else None)
        if key not in GENDER_CODES:
            raise ValueError(f"알 수 없는 성별 값: {v!r} (0=여성, 1=남성 또는 female/male)")
        out.append(GENDER_CODES[key])
    return out


def age_group_of(age):
    """원본 보호 속성 연령대: 20-34 / 35-54 / 55+ (20세 미만은 20-34)."""
    if age < 35:
        return '20-34'
    if age < 55:
        return '35-54'
    return '55+'


def normalize_age_group(values=None, ages=None):
    """나이(ages)가 있으면 나이로 3구간을 만든다. 없으면 이미 3구간 문자열이어야 한다.
    예전 5구간 숫자(AgeGroup 0-4)는 35·55세 경계를 나눌 수 없어 받지 않는다."""
    if ages is not None:
        return [age_group_of(float(a)) for a in _list(ages)]
    out = []
    for v in _list(values):
        if v not in AGE_GROUPS:
            raise ValueError(f"연령대는 {AGE_GROUPS} 중 하나여야 합니다: {v!r} "
                             "(5구간 AgeGroup이면 원래 나이 age 를 넘겨 주세요)")
        out.append(v)
    return out


def approvals(p_default, threshold=None):
    """부도 확률 → 승인 여부(1=승인). 기준값보다 낮으면 승인."""
    t = default_threshold() if threshold is None else threshold
    return [1 if p < t else 0 for p in _list(p_default)]


def approval_rates(approved, groups):
    """집단별 승인율 {집단: 비율}."""
    total, count = {}, {}
    for a, g in zip(_list(approved), _list(groups)):
        total[g] = total.get(g, 0) + 1
        count[g] = count.get(g, 0) + a
    return {g: count[g] / total[g] for g in sorted(total)}


def _rate(approved, groups, member):
    picked = [a for a, g in zip(approved, groups) if member(g)]
    return (sum(picked) / len(picked)) if picked else None


def disparate_impact(approved, groups, protected, attribute='group'):
    """DI Ratio = 보호 집단 승인율 / 비보호 집단 승인율.
    0.8 미만 또는 1.25 초과면 WARNING 로그를 남긴다."""
    approved, groups = _list(approved), _list(groups)
    p = _rate(approved, groups, lambda g: g == protected)
    r = _rate(approved, groups, lambda g: g != protected)
    if p is None or r is None:
        raise ValueError(f"{attribute}: 보호 집단({protected})과 비보호 집단이 모두 있어야 합니다")
    if r == 0:
        raise ValueError(f"{attribute}: 비보호 집단 승인율이 0이라 DI Ratio를 계산할 수 없습니다")
    di = p / r
    if di < DI_MIN or di > DI_MAX:
        logger.warning(f"[공정성 경고] {attribute} DI Ratio {di:.3f} (보호 집단 {protected}) - "
                       f"기준 {DI_MIN} 이상 {DI_MAX} 이하를 벗어났습니다")
    return di


def equalized_odds(y_true, approved, groups, protected):
    """Equalized Odds: 보호 집단과 비보호 집단의 TPR 차이, FPR 차이 (절댓값).
    y_true 는 연체 여부(1=연체). 승인을 좋은 결과로 본다."""
    y, a, g = _list(y_true), _list(approved), _list(groups)

    def rates(member):
        good = [ai for yi, ai, gi in zip(y, a, g) if member(gi) and yi == 0]
        bad = [ai for yi, ai, gi in zip(y, a, g) if member(gi) and yi == 1]
        tpr = sum(good) / len(good) if good else 0.0
        fpr = sum(bad) / len(bad) if bad else 0.0
        return tpr, fpr

    tpr_p, fpr_p = rates(lambda x: x == protected)
    tpr_r, fpr_r = rates(lambda x: x != protected)
    tpr_diff, fpr_diff = abs(tpr_p - tpr_r), abs(fpr_p - fpr_r)
    return {
        'tpr_diff': tpr_diff, 'fpr_diff': fpr_diff,
        'tpr': {'protected': tpr_p, 'reference': tpr_r},
        'fpr': {'protected': fpr_p, 'reference': fpr_r},
        'passed': tpr_diff <= EO_MAX and fpr_diff <= EO_MAX,
    }


def attribute_report(y_true, approved, groups, attribute):
    protected = PROTECTED[attribute]
    di = disparate_impact(approved, groups, protected, attribute)
    eo = equalized_odds(y_true, approved, groups, protected)
    return {
        'protected_group': protected,
        'approval_rates': approval_rates(approved, groups),
        'di_ratio': di,
        'di_passed': DI_MIN <= di <= DI_MAX,
        'equalized_odds_tpr_diff': eo['tpr_diff'],
        'equalized_odds_fpr_diff': eo['fpr_diff'],
        'equalized_odds_passed': eo['passed'],
    }


def fairness_report(y_true, p_default, gender=None, age_group=None, threshold=None):
    """/fairness 응답 형식. 대시보드 mock(src/dashboard/mock/fairness/fairness.json)과 같은 키를 맨 위에 둔다.
    맨 위 값은 기준에서 가장 멀리 벗어난 속성(가장 나쁜 쪽)의 값이다."""
    t = default_threshold() if threshold is None else threshold
    approved = approvals(p_default, t)
    by_attr = {}
    if gender is not None:
        by_attr['gender'] = attribute_report(y_true, approved, gender, 'gender')
    if age_group is not None:
        by_attr['age_group'] = attribute_report(y_true, approved, age_group, 'age_group')
    if not by_attr:
        raise ValueError("gender 또는 age_group 중 하나는 있어야 합니다")

    def distance(di):
        return max(DI_MIN - di, di - DI_MAX, 0) if di > 0 else float('inf')

    worst = max(by_attr.values(), key=lambda r: (distance(r['di_ratio']), abs(1 - r['di_ratio'])))
    return {
        'di_ratio': worst['di_ratio'],
        'equalized_odds_tpr_diff': max(r['equalized_odds_tpr_diff'] for r in by_attr.values()),
        'equalized_odds_fpr_diff': max(r['equalized_odds_fpr_diff'] for r in by_attr.values()),
        'threshold': t,
        'n': len(approved),
        'criteria': {'di_min': DI_MIN, 'di_max': DI_MAX, 'eo_max': EO_MAX},
        'by_attribute': by_attr,
    }
