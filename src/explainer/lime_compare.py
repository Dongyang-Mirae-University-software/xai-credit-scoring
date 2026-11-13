"""
보너스 과제 1: LIME 비교 분석.

과제 노션 원문
- LIME을 사용하여 개별 예측 설명을 생성하고, 동일 샘플에 대한 SHAP 결과와 비교한다.
- 두 기법의 상위 5개 중요 변수 일치율을 계산한다.

방법
- 같은 고객(같은 전처리 값)에 대해 SHAP(TreeExplainer, 부도 쪽 log-odds)과
  LIME(LimeTabularExplainer, 부도 확률 주변의 국소 선형 모형)을 각각 구한다.
- 고객마다 |기여도|가 큰 순서로 상위 5개 변수를 뽑아, 겹치는 수 ÷ 5 를 일치율로 본다.
- 겹친 변수 중 두 기법이 같은 방향(부도 쪽 + / 정상 쪽 -)으로 본 비율도 함께 센다.
- LIME 은 무작위 표본으로 설명을 만들므로, 같은 고객을 다른 시드로 다시 돌려
  LIME 자기 자신과의 일치율(안정성)도 잰다. SHAP(TreeExplainer)은 같은 입력이면 항상 같은 값이다.

lime 패키지가 필요하다 (requirements.txt). 무거운 라이브러리라 함수 안에서 불러온다.
"""

TOP_K = 5
LIME_NUM_SAMPLES = 5000


def top_k_features(contributions, columns, k=TOP_K):
    """|기여도|가 큰 순서로 상위 k개 변수 이름. 기여도가 같으면 columns 순서를 따른다."""
    order = sorted(range(len(columns)), key=lambda i: (-abs(float(contributions[i])), i))
    return [columns[i] for i in order[:k]]


def top_k_agreement(a, b, k=TOP_K):
    """두 상위 k개 목록이 겹치는 비율 (0 ~ 1). 순서는 보지 않는다."""
    if k <= 0:
        raise ValueError("k 는 1 이상이어야 합니다.")
    return len(set(a[:k]) & set(b[:k])) / k


def sign_agreement(shap_row, lime_row, shared, columns):
    """겹친 변수 중 두 기법의 기여 방향(부호)이 같은 비율. 겹친 변수가 없으면 None."""
    if not shared:
        return None
    idx = {c: i for i, c in enumerate(columns)}
    same = sum(1 for f in shared if (shap_row[idx[f]] > 0) == (lime_row[idx[f]] > 0))
    return same / len(shared)


def make_lime_explainer(train_X, random_state=42):
    """Train 전처리 값으로 LIME 설명기를 만든다 (분포·구간 나누기에 Train만 쓴다)."""
    from lime.lime_tabular import LimeTabularExplainer

    return LimeTabularExplainer(
        training_data=train_X.values,
        feature_names=list(train_X.columns),
        class_names=['정상', '부도'],
        mode='classification',
        discretize_continuous=True,
        random_state=random_state,
    )


def lime_contributions(explainer, predict_proba, x_row, n_features, num_samples=LIME_NUM_SAMPLES):
    """고객 한 명의 LIME 가중치 (변수 순서대로, 부도 쪽 +)."""
    import numpy as np

    exp = explainer.explain_instance(np.asarray(x_row, dtype=float), predict_proba,
                                     labels=(1,), num_features=n_features, num_samples=num_samples)
    weights = np.zeros(n_features)
    for i, w in exp.as_map()[1]:
        weights[i] = w
    return weights


def compare(shap_values, lime_values, columns, k=TOP_K):
    """
    고객별 SHAP · LIME 상위 k개 비교.
    shap_values, lime_values: (사람 수 × 변수 수). 돌려주는 값: 고객별 기록 목록.
    """
    rows = []
    for s, l in zip(shap_values, lime_values):
        s_top, l_top = top_k_features(s, columns, k), top_k_features(l, columns, k)
        shared = [f for f in s_top if f in l_top]
        rows.append({
            'shap_top': s_top, 'lime_top': l_top, 'shared': shared,
            'agreement': top_k_agreement(s_top, l_top, k),
            'sign_agreement': sign_agreement(s, l, shared, columns),
            'top1_same': s_top[0] == l_top[0],
        })
    return rows


def summarize(rows, k=TOP_K):
    """비교 기록 → 평균 일치율, 겹친 개수 분포, 방향 일치율, 1위 일치율."""
    import numpy as np

    if not rows:
        return {'n': 0}
    agree = np.array([r['agreement'] for r in rows])
    signs = [r['sign_agreement'] for r in rows if r['sign_agreement'] is not None]
    counts = {str(c): int(sum(1 for r in rows if len(r['shared']) == c)) for c in range(k + 1)}
    return {
        'n': len(rows),
        'mean_agreement': float(agree.mean()),
        'std_agreement': float(agree.std(ddof=1)) if len(rows) > 1 else 0.0,
        'median_agreement': float(np.median(agree)),
        'overlap_count_distribution': counts,
        'mean_sign_agreement': float(np.mean(signs)) if signs else None,
        'top1_same_rate': float(np.mean([r['top1_same'] for r in rows])),
    }


def top_k_frequency(rows, key, columns):
    """각 변수가 고객별 상위 k개에 든 비율 (key: 'shap_top' 또는 'lime_top')."""
    n = len(rows) or 1
    return {c: sum(1 for r in rows if c in r[key]) / n for c in columns}
