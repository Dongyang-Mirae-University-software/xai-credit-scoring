"""
Fairlearn으로 공정성 지표를 다시 계산해 우리 함수(metrics.py) 값과 맞는지 확인한다.

원본 과제 기준
- 「AIF360/Fairlearn 기반 공정성 검증 및 Bias Mitigation 모듈을 구현한다.」
  → 지표 교차 확인(cross_check)과, Threshold Adjustment 를 Fairlearn ThresholdOptimizer 와 비교(threshold_optimizer_report)
- 개발 환경 「공정성 라이브러리: Fairlearn 또는 AIF360」

metrics.py 의 정의를 Fairlearn 용어로 바꾸면 다음과 같다 (승인을 좋은 결과로 본다).
- 정답 y_good = 1 - 연체 여부, 예측 = 승인 여부(1=승인)
- 승인율 = selection_rate
- TPR (정상 상환자 중 승인 비율) = true_positive_rate
- FPR (연체자 중 승인 비율) = false_positive_rate
- DI Ratio = 보호 집단 selection_rate / 비보호 집단 selection_rate
  (Fairlearn 의 demographic_parity_ratio 는 '작은 값 / 큰 값'이라 보호 집단 방향이 없어 쓰지 않는다)
- Equalized Odds 차이 = 보호 집단과 비보호 집단의 TPR·FPR 차이.
  두 집단일 때 Fairlearn equalized_odds_difference 는 두 차이 중 큰 값과 같다.
"""

import logging

from src.fairness import metrics as fm

logger = logging.getLogger(__name__)

TOLERANCE = 1e-9


def _fairlearn():
    try:
        from fairlearn import metrics as flm
    except ImportError as e:
        raise ImportError("fairlearn 이 없습니다. pip install -r requirements.txt 로 설치하세요") from e
    return flm


def fairlearn_attribute(y_true, approved, groups, protected):
    """Fairlearn MetricFrame 으로 보호 집단 / 비보호 집단의 승인율·TPR·FPR, DI, EO 를 계산한다."""
    flm = _fairlearn()
    y = [1 - int(v) for v in fm._list(y_true)]
    a = [int(v) for v in fm._list(approved)]
    side = ['protected' if g == protected else 'reference' for g in fm._list(groups)]
    frame = flm.MetricFrame(
        metrics={'selection_rate': flm.selection_rate,
                 'tpr': flm.true_positive_rate,
                 'fpr': flm.false_positive_rate},
        y_true=y, y_pred=a, sensitive_features=side)
    g = frame.by_group
    p, r = g.loc['protected'], g.loc['reference']
    return {
        'di_ratio': float(p['selection_rate'] / r['selection_rate']),
        'equalized_odds_tpr_diff': float(abs(p['tpr'] - r['tpr'])),
        'equalized_odds_fpr_diff': float(abs(p['fpr'] - r['fpr'])),
        'equalized_odds_difference': float(flm.equalized_odds_difference(y, a, sensitive_features=side)),
        'approval_rates': {str(k): float(v) for k, v in
                           flm.MetricFrame(metrics=flm.selection_rate, y_true=y, y_pred=a,
                                           sensitive_features=fm._list(groups)).by_group.items()},
    }


def cross_check(report, y_true, p_default, gender=None, age_group=None):
    """fairness_report() 결과(report)를 Fairlearn 값과 비교한다.
    속성별로 우리 값, Fairlearn 값, 일치 여부를 돌려준다. 하나라도 다르면 WARNING 로그."""
    approved = fm.approvals(p_default, report['threshold'])
    groups_by_attr = {'gender': gender, 'age_group': age_group}
    out = {'library': 'fairlearn', 'tolerance': TOLERANCE, 'by_attribute': {}}
    try:
        import fairlearn
        out['library_version'] = fairlearn.__version__
    except ImportError:
        pass
    all_match = True
    for attr, ours in report['by_attribute'].items():
        fl = fairlearn_attribute(y_true, approved, groups_by_attr[attr], ours['protected_group'])
        checks = {
            'di_ratio': (ours['di_ratio'], fl['di_ratio']),
            'equalized_odds_tpr_diff': (ours['equalized_odds_tpr_diff'], fl['equalized_odds_tpr_diff']),
            'equalized_odds_fpr_diff': (ours['equalized_odds_fpr_diff'], fl['equalized_odds_fpr_diff']),
            'equalized_odds_difference': (max(ours['equalized_odds_tpr_diff'], ours['equalized_odds_fpr_diff']),
                                          fl['equalized_odds_difference']),
        }
        for g, rate in ours['approval_rates'].items():
            checks[f'approval_rate[{g}]'] = (rate, fl['approval_rates'][g])
        rows = {k: {'ours': o, 'fairlearn': f, 'match': abs(o - f) <= TOLERANCE} for k, (o, f) in checks.items()}
        matched = all(r['match'] for r in rows.values())
        all_match &= matched
        out['by_attribute'][attr] = {'protected_group': ours['protected_group'], 'match': matched, 'values': rows}
        if not matched:
            bad = [k for k, r in rows.items() if not r['match']]
            logger.warning(f"[공정성 교차 확인] {attr}: Fairlearn 값과 다름 → {bad}")
    out['match'] = all_match
    return out


# ---------------------------------------------------------------------------
# Fairlearn ThresholdOptimizer: 우리 Threshold Adjustment(mitigation.py)와 같은 조건으로 비교
# - 집단: 성별 × 연령대 6개 (우리 Threshold Adjustment 와 같음)
# - 기준값은 Validation 으로 정하고 Test 에는 적용만 한다 (원본 「테스트 데이터의 라벨을 학습에 사용하는 것을 금지한다」)
# - constraints='demographic_parity': 집단별 승인율을 같게 (우리 Threshold Adjustment 와 같은 목표)
#   constraints='equalized_odds'   : 집단별 TPR·FPR 을 같게
# - objective 기본값은 balanced_accuracy_score. 연체자가 6.7%뿐이라 accuracy_score 로 고르면 거의 모두 승인하는 쪽이
#   정확도가 높아져 연체자 승인 비율이 크게 늘어난다 (최종 모델이 Recall 을 중시한 것과 맞지 않음).
# - 경계에서는 기준값 두 개를 무작위로 섞으므로 random_state 를 고정한다.
# ---------------------------------------------------------------------------

try:
    from sklearn.base import BaseEstimator, ClassifierMixin
except ImportError:  # sklearn 은 requirements 에 있으므로 보통 여기로 오지 않는다
    BaseEstimator = ClassifierMixin = object


class _ApprovalScore(ClassifierMixin, BaseEstimator):
    """이미 학습된 모델의 부도 확률을 Fairlearn 에 넘기기 위한 껍데기.
    입력 X 의 첫 칸이 부도 확률이고, predict_proba[:, 1] 은 '정상 상환(승인이 맞음)' 확률 = 1 - 부도 확률."""
    classes_ = [0, 1]

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None, **kwargs):
        return self

    def predict_proba(self, X):
        import numpy as np
        p = np.asarray(X, dtype=float).reshape(len(X), -1)[:, 0]
        return np.column_stack([p, 1 - p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def threshold_optimizer_report(fit_y, fit_p, fit_gender, fit_age_group,
                               y_true, p_default, gender, age_group,
                               constraints='demographic_parity', objective='balanced_accuracy_score',
                               random_state=42, threshold=None):
    """Fairlearn ThresholdOptimizer 적용 전/후 공정성 비교 (mitigation_report 와 같은 before/after 키)."""
    import numpy as np
    from fairlearn.postprocessing import ThresholdOptimizer
    from src.fairness import mitigation as mt

    t = fm.default_threshold() if threshold is None else threshold
    fit_groups = mt.combine_groups(fit_gender, fit_age_group)
    groups = mt.combine_groups(gender, age_group)
    to = ThresholdOptimizer(estimator=_ApprovalScore(), constraints=constraints, objective=objective, prefit=True,
                            predict_method='predict_proba')
    to.fit(np.asarray(fm._list(fit_p)).reshape(-1, 1), [1 - int(v) for v in fm._list(fit_y)],
           sensitive_features=fit_groups)
    approved = [int(a) for a in to.predict(np.asarray(fm._list(p_default)).reshape(-1, 1),
                                           sensitive_features=groups, random_state=random_state)]

    y = [int(v) for v in fm._list(y_true)]
    before_approved = fm.approvals(p_default, t)
    before = fm.fairness_report(y_true, p_default, gender=gender, age_group=age_group, threshold=t)
    after = mt._report_from_approved(y_true, approved, gender, age_group)

    def overall(a):
        bad = [ai for ai, yi in zip(a, y) if yi == 1]
        good = [ai for ai, yi in zip(a, y) if yi == 0]
        return {'approval_rate': sum(a) / len(a),
                'defaulters_approved_rate': sum(bad) / len(bad) if bad else None,
                'good_customers_approved_rate': sum(good) / len(good) if good else None,
                'accuracy': sum(1 for ai, yi in zip(a, y) if ai == 1 - yi) / len(y)}

    try:
        import fairlearn
        version = fairlearn.__version__
    except ImportError:
        version = None
    return {
        'method': f'fairlearn_threshold_optimizer_{constraints}',
        'library': 'fairlearn', 'library_version': version,
        'constraints': constraints, 'objective': objective, 'groups': 'gender|age_group',
        'threshold': t, 'threshold_fit_on': 'validation', 'random_state': random_state,
        'before': mt._summary(before, mt._performance(y_true, p_default, before_approved)),
        'after': mt._summary(after, mt._performance(y_true, p_default, approved)),
        'overall': {'before': overall(before_approved), 'after': overall(approved)},
    }
