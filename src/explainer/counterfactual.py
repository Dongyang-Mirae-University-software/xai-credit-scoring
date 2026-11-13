"""
보너스 과제 2: Counterfactual Explanation (반사실적 설명).

과제 노션 원문
- 거절된 고객에게 "어떤 조건이 바뀌면 승인될 수 있는지"를 제시하는 Counterfactual Explanation 기능을 구현한다.

방법 (원래 단위에서 탐색, 모델 호출만 사용하므로 모델 종류와 상관없음)
1. 바꿀 수 있는 칸만 움직인다 (ACTIONABLE). 나이 · 과거 연체 횟수 · 부양가족 수처럼
   본인이 바꿀 수 없거나 이미 지난 기록은 움직이지 않는다 (IMMUTABLE).
2. 칸마다 좋아지는 방향으로만, Train 분포 안(1% ~ 99% 분위수)에서만 움직인다.
   데이터에 거의 없는 값(예: 한도 사용 비율 -10%)을 제안하지 않기 위해서다.
3. 한 칸만 바꾸기 (single): 칸마다 현재 값에서 한계까지 STEPS 단계로 나눠, 승인되는 가장 작은 변화를 찾는다.
4. 여러 칸 함께 바꾸기 (combined): 한 단계씩, 부도 확률을 가장 많이 낮추는 칸을 고르는 것을
   승인될 때까지 반복한다 (탐욕 탐색). 그 뒤 바꾼 칸을 하나씩 원래 쪽으로 되돌려 보며,
   승인이 유지되면 되돌린다 (불필요한 변화 줄이기).

predict_proba(DataFrame[원래 단위 15칸]) → 부도 확률 배열 을 받는다. 스케일링은 호출하는 쪽이 한다.
"""

from src.explainer.feature_names import korean_name

STEPS = 20

# 칸: (좋아지는 방향 +1 증가 / -1 감소, Train 1% 분위수, Train 99% 분위수, 최대 상대 변화)
# 분위수는 train_processed.csv 를 preprocess_scaler_v1.0 으로 되돌린 값 (main_bonus_counterfactual.py 가 다시 확인)
ACTIONABLE = {
    'RevolvingUtilizationOfUnsecuredLines': (-1, 0.0, 1.0953, None),   # 한도 사용 비율 낮추기
    'DebtRatio': (-1, 0.0, 264.5, None),                               # 부채 비율 낮추기
    'MonthlyIncome': (+1, 0.0, 22900.0, 0.5),                          # 월소득: 현재의 +50% 까지만
    'telecom_payment_rate': (+1, 0.7483, 1.0, None),
    'utility_payment_rate': (+1, 0.6463, 1.0, None),
    'spending_consistency': (+1, 82.9843, 95.1817, None),
    'regular_payment_count': (+1, 10.0, 13.0, None),
    'app_login_frequency': (+1, 18.0, 23.0, None),
}
INTEGER_FEATURES = {'regular_payment_count', 'app_login_frequency'}
# 사람에게 보여 줄 때의 (배수, 단위). 월소득 모델 단위 1 = 1,000원 (src/api/model.py INCOME_KRW_PER_UNIT 기본값)
DISPLAY_UNITS = {
    'RevolvingUtilizationOfUnsecuredLines': (100, '%'), 'DebtRatio': (100, '%'), 'MonthlyIncome': (1000, '원'),
    'telecom_payment_rate': (100, '%'), 'utility_payment_rate': (100, '%'), 'spending_consistency': (1, '점'),
    'regular_payment_count': (1, '건'), 'app_login_frequency': (1, '회'),
}
IMMUTABLE = [
    'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate', 'NumberOfDependents', 'NumberRealEstateLoansOrLines',
    'NumberOfOpenCreditLinesAndLoans',
]


def target_value(feature, current):
    """이 칸을 가장 많이 좋게 움직였을 때의 값. 이미 한계를 넘어 있으면 현재 값 (움직일 수 없음)."""
    direction, low, high, max_rel = ACTIONABLE[feature]
    if direction > 0:
        limit = high if max_rel is None else min(high, current * (1 + max_rel)) if current > 0 else high
        return max(current, limit)
    return min(current, low)


def levels(feature, current, steps=STEPS):
    """현재 값에서 한계까지 steps 단계 (현재 값 제외, 가까운 순서). 정수 칸은 겹치는 값을 뺀다."""
    end = target_value(feature, current)
    if end == current:
        return []
    vals = [current + (end - current) * i / steps for i in range(1, steps + 1)]
    if feature in INTEGER_FEATURES:
        out = []
        for v in vals:
            r = float(round(v))
            if r != current and r not in out:
                out.append(r)
        return out
    return vals


def _batch(base, changes_list):
    import pandas as pd
    rows = []
    for changes in changes_list:
        row = dict(base)
        row.update(changes)
        rows.append(row)
    return pd.DataFrame(rows)


def single_changes(x, predict_proba, threshold, steps=STEPS):
    """칸 하나만 바꿔 승인되는 가장 작은 변화. 칸마다 {feature, suggested, new_probability} 또는 None(불가능)."""
    import numpy as np
    out = {}
    for f in ACTIONABLE:
        cand = levels(f, float(x[f]), steps)
        if not cand:
            out[f] = None
            continue
        p = np.asarray(predict_proba(_batch(x, [{f: v} for v in cand])))
        ok = np.where(p < threshold)[0]
        out[f] = None if len(ok) == 0 else {'feature': f, 'suggested': cand[ok[0]], 'new_probability': float(p[ok[0]])}
    return out


def combined_changes(x, predict_proba, threshold, steps=STEPS, max_iter=200):
    """
    여러 칸을 함께 바꾸는 탐욕 탐색. 돌려주는 값:
    {'success': bool, 'changes': {칸: 제안 값}, 'new_probability': float, 'iterations': int}
    """
    import numpy as np
    current = {f: float(x[f]) for f in ACTIONABLE}
    grid = {f: levels(f, current[f], steps) for f in ACTIONABLE}
    pos = {f: 0 for f in ACTIONABLE}            # grid 안에서 몇 단계 움직였나 (0 = 그대로)
    changes = {}
    prob = float(np.asarray(predict_proba(_batch(x, [{}])))[0])
    it = 0
    while prob >= threshold and it < max_iter:
        movable = [f for f in ACTIONABLE if pos[f] < len(grid[f])]
        if not movable:
            break
        trials = [{**changes, f: grid[f][pos[f]]} for f in movable]
        p = np.asarray(predict_proba(_batch(x, trials)))
        best = int(np.argmin(p))
        f = movable[best]
        changes[f] = grid[f][pos[f]]
        pos[f] += 1
        prob = float(p[best])
        it += 1
    success = prob < threshold
    if success:
        changes, prob = _prune(x, changes, grid, pos, predict_proba, threshold, prob)
    return {'success': success, 'changes': changes, 'new_probability': prob, 'iterations': it}


def _prune(x, changes, grid, pos, predict_proba, threshold, prob):
    """바꾼 칸을 한 단계씩 원래 쪽으로 되돌려 보고, 승인이 유지되면 되돌린다."""
    import numpy as np
    improved = True
    while improved:
        improved = False
        for f in list(changes):
            k = pos[f] - 1                       # 지금 grid[f][k] 에 있다
            trial = dict(changes)
            if k == 0:
                trial.pop(f)
            else:
                trial[f] = grid[f][k - 1]
            p = float(np.asarray(predict_proba(_batch(x, [trial])))[0])
            if p < threshold:
                changes, prob = trial, p
                pos[f] -= 1
                improved = True
    return changes, prob


def describe(feature, current, suggested, unit_scale=None):
    """사람이 읽는 한 줄. unit_scale: 칸별 (표시 배수, 단위 문자열). 없으면 원래 값 그대로."""
    mult, unit = (unit_scale or {}).get(feature, (1, ''))
    verb = '낮추기' if suggested < current else '높이기'
    fmt = (lambda v: f"{v * mult:,.0f}{unit}") if unit in ('원', '건', '회', '점') else (lambda v: f"{v * mult:.1f}{unit}")
    return f"{korean_name(feature)} {fmt(current)} → {fmt(suggested)} ({verb})"


def normalized_distance(x, changes):
    """제안의 크기: 칸마다 |변화| ÷ (Train 99% - 1% 분위수) 의 합. 칸 사이 단위를 맞추기 위한 값."""
    return float(sum(abs(v - float(x[f])) / (ACTIONABLE[f][2] - ACTIONABLE[f][1]) for f, v in changes.items()))
