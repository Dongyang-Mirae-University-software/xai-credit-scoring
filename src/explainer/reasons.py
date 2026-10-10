"""
거절 사유 Top 5 만들기 (SHAP 값 → 사람이 읽는 문장).

원본 과제 형식
- 각 거절 사유에 feature, current_value, shap_contribution, explanation 4개 필드
- 문장 예시: "통신비 정상납부율이 78%로, 동일 연령대 평균(92%) 대비 낮습니다 (기여도: -0.15)"
- 거절 시 최소 3가지 이상 주요 사유를 제시

부호 약속
- 모델은 '부도 확률'을 예측하므로 SHAP 값이 양수면 부도 쪽(거절 쪽)으로 민 것이다.
- 원본 예시처럼 기여도는 '승인 쪽' 기준으로 적는다: shap_contribution = -(부도 쪽 SHAP 값).
  그래서 거절 사유의 기여도는 음수로 나온다.

이 파일은 numpy·shap 없이 동작한다 (SHAP 값 계산은 shap_explainer.py 가 맡는다).
"""

from src.explainer.feature_names import korean_name, format_value

DEFAULT_TOP_K = 5


def age_group_of(age):
    """원본 보호 속성 연령대: 20-34 / 35-54 / 55+."""
    if age is None:
        return None
    if age < 35:
        return '20-34'
    if age < 55:
        return '35-54'
    return '55+'


def reference_values(rows, features, group_key='age_group'):
    """
    '동일 연령대 평균'을 만든다. rows: dict 목록(학습 데이터, 크기 맞추기 전 원래 값).
    반환: {연령대: {feature: 평균}} 와 전체 평균 '_all'.
    """
    sums, counts = {}, {}
    for r in rows:
        group = r.get(group_key) or age_group_of(r.get('age'))
        for key in (group, '_all'):
            if key is None:
                continue
            s = sums.setdefault(key, {})
            c = counts.setdefault(key, {})
            for f in features:
                v = r.get(f)
                if v is None or v != v:  # 빈칸(None, NaN)은 건너뛴다
                    continue
                s[f] = s.get(f, 0.0) + float(v)
                c[f] = c.get(f, 0) + 1
    return {g: {f: sums[g][f] / counts[g][f] for f in sums[g]} for g in sums}


def _has_final(text, rieul_ok=False):
    """마지막 글자에 받침이 있는지. 숫자는 읽는 소리로 판단한다 (예: 3=삼 → 받침 있음)."""
    if not text:
        return False
    ch = text[-1]
    if '가' <= ch <= '힣':
        final = (ord(ch) - ord('가')) % 28
        return final != 0 and not (rieul_ok and final == 8)  # 8 = ㄹ
    digit_final = {'0': True, '1': True, '3': True, '6': True, '7': True, '8': True}
    if ch in digit_final:
        return not (rieul_ok and ch in '178')  # 일·칠·팔은 ㄹ 받침
    return False


def _subject(name):
    return name + ('이' if _has_final(name) else '가')


def _with_ro(value_text):
    return value_text + ('으로' if _has_final(value_text, rieul_ok=True) else '로')


def _sentence(feature, current, reference, contribution, group_label):
    name = korean_name(feature)
    cur = format_value(feature, current)
    if reference is None:
        return f"{_subject(name)} {cur}입니다 (기여도: {contribution:+.2f})"
    ref = format_value(feature, reference)
    direction = '낮습니다' if current < reference else '높습니다'
    if current == reference:
        direction = '같습니다'
    return f"{_subject(name)} {_with_ro(cur)}, {group_label} 평균({ref}) 대비 {direction} (기여도: {contribution:+.2f})"


def build_rejection_reasons(feature_values, shap_values, references=None, age=None,
                            top_k=DEFAULT_TOP_K, only_negative=True):
    """
    고객 한 명의 거절 사유를 만든다.

    feature_values: {feature: 원래 값} (크기 맞추기 전 값이어야 문장이 자연스럽다)
    shap_values:    {feature: 부도 쪽 SHAP 값}
    references:     reference_values() 결과. 없으면 평균 비교 없이 문장을 만든다.
    age:            고객 나이. 주면 같은 연령대 평균과 비교한다.
    only_negative:  True 면 승인에 불리한(기여도 < 0) 항목만 고른다.
    """
    group = age_group_of(age) if age is not None else None
    ref_table = {}
    group_label = '전체'
    if references:
        if group and group in references:
            ref_table = references[group]
            group_label = '동일 연령대'
        else:
            ref_table = references.get('_all', {})

    items = []
    for feature, value in shap_values.items():
        contribution = -float(value)  # 승인 쪽 기준
        if only_negative and contribution >= 0:
            continue
        items.append((contribution, feature))
    items.sort()  # 가장 불리한(가장 작은) 것부터

    reasons = []
    for rank, (contribution, feature) in enumerate(items[:top_k], start=1):
        current = feature_values.get(feature)
        reference = ref_table.get(feature)
        reasons.append({
            'rank': rank,
            'feature': feature,
            'feature_name_kr': korean_name(feature),
            'current_value': current,
            'reference_value': reference,
            'shap_contribution': round(contribution, 4),
            'explanation': _sentence(feature, current, reference, contribution, group_label),
        })
    return reasons
