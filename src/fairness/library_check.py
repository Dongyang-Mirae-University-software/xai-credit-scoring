"""
Fairlearn으로 공정성 지표를 다시 계산해 우리 함수(metrics.py) 값과 맞는지 확인한다.

원본 과제 기준
- 「AIF360/Fairlearn 기반 공정성 검증 및 Bias Mitigation 모듈을 구현한다.」
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
