"""
Bias Mitigation: Reweighting(학습 전)과 Threshold Adjustment(학습 후), 적용 전/후 비교.

원본 과제 기준
- 「Bias Mitigation 기법이 1개 이상 구현되어 있다 (Reweighting 또는 Threshold Adjustment).」
- 「Mitigation 적용 전/후 공정성 지표 비교 표가 리포트에 포함되어 있다.」
- 「`/fairness/mitigation` API 호출 시 Mitigation 전/후 비교 결과가 JSON으로 반환된다.」

두 방법
- Reweighting: (집단, 연체 여부) 조합마다 학습 가중치를 준다. w = P(집단) × P(연체 여부) / P(집단, 연체 여부).
  가중치를 주면 모든 집단에서 연체 비율이 같아진 것처럼 학습된다. 모델을 다시 학습해야 한다.
  SMOTE처럼 행 수를 바꾸는 파이프라인에는 쓸 수 없다 (class_weight / scale_pos_weight 모델에 사용).
- Threshold Adjustment: 집단마다 승인 기준값을 따로 정해, 모든 집단의 승인율을 전체 승인율에 맞춘다.
  모델을 다시 학습하지 않는다.
"""

import logging

from src.fairness.metrics import _list, approvals, default_threshold, fairness_report


def combine_groups(*columns):
    """여러 보호 속성을 하나의 집단 이름으로 합친다. 예: ('female', '55+') → 'female|55+'."""
    cols = [_list(c) for c in columns if c is not None]
    return ['|'.join(map(str, parts)) for parts in zip(*cols)]


def reweighing_weights(y_true, groups):
    """Reweighting 가중치 (Kamiran & Calders). 평균은 1이 된다."""
    y, g = _list(y_true), _list(groups)
    n = len(y)
    n_g, n_y, n_gy = {}, {}, {}
    for yi, gi in zip(y, g):
        n_g[gi] = n_g.get(gi, 0) + 1
        n_y[yi] = n_y.get(yi, 0) + 1
        n_gy[(gi, yi)] = n_gy.get((gi, yi), 0) + 1
    return [(n_g[gi] * n_y[yi]) / (n * n_gy[(gi, yi)]) for yi, gi in zip(y, g)]


def fit_with_weights(model, X, y, sample_weight):
    """같은 설정의 새 모델을 가중치로 학습한다. Pipeline이면 마지막 단계에 가중치를 넘긴다."""
    from sklearn.base import clone
    m = clone(model)
    steps = getattr(m, 'steps', None)
    if steps:
        names = [type(s).__name__ for _, s in steps]
        if any('SMOTE' in nm or 'Sampler' in nm for nm in names):
            raise ValueError("SMOTE 같은 샘플링 단계가 있는 모델에는 Reweighting을 쓸 수 없습니다")
        m.fit(X, y, **{f'{steps[-1][0]}__sample_weight': sample_weight})
    else:
        m.fit(X, y, sample_weight=sample_weight)
    return m


def _quantile(sorted_values, q):
    if not sorted_values:
        return None
    pos = min(max(q, 0.0), 1.0) * len(sorted_values)
    i = int(pos)
    if i >= len(sorted_values):
        return sorted_values[-1] + 1e-12
    return sorted_values[i]


def group_thresholds(p_default, groups, threshold=None):
    """집단별 기준값: 각 집단의 승인율이 전체 승인율(같은 기준값일 때)과 같아지도록 정한다."""
    t = default_threshold() if threshold is None else threshold
    p, g = _list(p_default), _list(groups)
    target = sum(approvals(p, t)) / len(p)
    by_group = {}
    for pi, gi in zip(p, g):
        by_group.setdefault(gi, []).append(pi)
    # 승인 = p < 기준값 이므로, 집단 안에서 아래에서 target 비율 지점의 값이 그 집단의 기준값
    return {gi: _quantile(sorted(vals), target) for gi, vals in by_group.items()}, target


def approvals_with_thresholds(p_default, groups, thresholds, default=None):
    """집단별 기준값으로 승인 여부. 기준값을 정할 때 없던 집단은 공통 기준값(default)을 쓴다."""
    t = default_threshold() if default is None else default
    return [1 if pi < thresholds.get(gi, t) else 0 for pi, gi in zip(_list(p_default), _list(groups))]


def _performance(y_true, p_default, approved):
    """전/후 표에 공정성과 함께 보일 성능: AUC·KS(부도 확률 기준)와 전체 승인율."""
    from src.explainer.performance import ks_statistic, roc_auc
    approved = _list(approved)
    return {'auc': roc_auc(y_true, p_default), 'ks': ks_statistic(y_true, p_default),
            'approval_rate': sum(approved) / len(approved)}


def _summary(report, performance=None):
    keys = ('di_ratio', 'equalized_odds_tpr_diff', 'equalized_odds_fpr_diff')
    out = {k: report[k] for k in keys}
    out['by_attribute'] = report['by_attribute']
    out.update(performance or {})
    return out


def _report_from_approved(y_true, approved, gender, age_group):
    # fairness_report 는 확률을 받으므로, 승인(1)을 확률 0, 거절(0)을 확률 1로 넘겨 같은 계산을 쓴다
    pseudo = [0.0 if a else 1.0 for a in approved]
    return fairness_report(y_true, pseudo, gender=gender, age_group=age_group, threshold=0.5)


def mitigation_report(y_true, p_before, gender=None, age_group=None,
                      method='threshold_adjustment', p_after=None, threshold=None,
                      fit_p_default=None, fit_gender=None, fit_age_group=None):
    """/fairness/mitigation 응답 형식. 대시보드 mock(fairness_mitigation.json)과 같은 before/after 키.

    y_true, p_before, gender, age_group: 평가할 데이터(Test).
    method='threshold_adjustment': 집단별 기준값을 fit_* 데이터(Validation)로 정하고, 평가 데이터에는 적용만 한다.
        원본 「테스트 데이터의 라벨을 학습에 사용하는 것을 금지한다」와 'Test는 한 번만 평가' 원칙 때문.
        fit_* 를 주지 않으면 평가 데이터로 기준값을 정하고 경고를 남긴다(예시·테스트용).
    method='reweighting': 가중치로 다시 학습한 모델의 확률 p_after 를 함께 넘긴다."""
    t = default_threshold() if threshold is None else threshold
    before = fairness_report(y_true, p_before, gender=gender, age_group=age_group, threshold=t)
    result = {'method': method, 'threshold': t,
              'before': _summary(before, _performance(y_true, p_before, approvals(p_before, t)))}
    if method == 'threshold_adjustment':
        groups = combine_groups(gender, age_group)
        if fit_p_default is not None:
            fit_groups = combine_groups(fit_gender, fit_age_group)
            thresholds, target = group_thresholds(fit_p_default, fit_groups, t)
            result['threshold_fit_on'] = 'validation'
        else:
            logging.getLogger(__name__).warning(
                "Threshold Adjustment 기준값을 평가 데이터로 정합니다. 실제 평가에는 fit_p_default(Validation)를 넘기세요.")
            thresholds, target = group_thresholds(p_before, groups, t)
            result['threshold_fit_on'] = 'evaluation_data'
        approved = approvals_with_thresholds(p_before, groups, thresholds, default=t)
        # 모델은 그대로라 AUC·KS는 적용 전과 같고, 승인율만 바뀐다
        result['after'] = _summary(_report_from_approved(y_true, approved, gender, age_group),
                                   _performance(y_true, p_before, approved))
        result['group_thresholds'] = thresholds
        result['target_approval_rate'] = target
    elif method == 'reweighting':
        if p_after is None:
            raise ValueError("reweighting 은 가중치로 다시 학습한 모델의 확률 p_after 가 필요합니다")
        result['after'] = _summary(fairness_report(y_true, p_after, gender=gender, age_group=age_group, threshold=t),
                                   _performance(y_true, p_after, approvals(p_after, t)))
    else:
        raise ValueError(f"알 수 없는 방법: {method}")
    return result


def comparison_table(report):
    """리포트용 전/후 비교 표 (마크다운)."""
    rows = [('DI Ratio', 'di_ratio'),
            ('Equalized Odds TPR 차이', 'equalized_odds_tpr_diff'),
            ('Equalized Odds FPR 차이', 'equalized_odds_fpr_diff')]
    lines = [f"| 지표 | 적용 전 | 적용 후 ({report['method']}) |", "|---|---|---|"]
    for label, key in rows:
        lines.append(f"| {label} | {report['before'][key]:.3f} | {report['after'][key]:.3f} |")
    for attr, b in report['before']['by_attribute'].items():
        a = report['after']['by_attribute'][attr]
        for grp, rate in b['approval_rates'].items():
            lines.append(f"| 승인율 {attr}={grp} | {rate:.3f} | {a['approval_rates'][grp]:.3f} |")
    return '\n'.join(lines)
