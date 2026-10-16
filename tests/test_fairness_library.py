"""Fairlearn 교차 확인 테스트: 우리 공정성 함수 값이 Fairlearn 값과 같아야 한다."""
import random

import pytest

pytest.importorskip('fairlearn')

from src.fairness import metrics as fm
from src.fairness import library_check as lc


def _sample(n=2000, seed=42):
    rnd = random.Random(seed)
    gender = [rnd.choice(['female', 'male']) for _ in range(n)]
    age = [rnd.choice(['20-34', '35-54', '55+']) for _ in range(n)]
    y = [1 if rnd.random() < 0.1 else 0 for _ in range(n)]
    # 집단마다 확률을 조금씩 다르게 해 차이가 생기게 한다
    p = [min(0.99, max(0.01, rnd.random() * 0.6 + 0.3 * yi + (0.05 if g == 'female' else 0)
                       + (0.1 if a == '20-34' else 0)))
         for yi, g, a in zip(y, gender, age)]
    return y, p, gender, age


def test_fairlearn_matches_our_di_and_equalized_odds():
    y, p, gender, age = _sample()
    report = fm.fairness_report(y, p, gender=gender, age_group=age)
    check = lc.cross_check(report, y, p, gender=gender, age_group=age)
    assert check['match'] is True
    for attr in ('gender', 'age_group'):
        values = check['by_attribute'][attr]['values']
        assert values['di_ratio']['fairlearn'] == pytest.approx(report['by_attribute'][attr]['di_ratio'])
        assert values['equalized_odds_tpr_diff']['match'] and values['equalized_odds_fpr_diff']['match']


def test_fairlearn_known_values():
    # test_fairness.py 의 손 계산 예시와 같은 값: TPR 0.75 vs 0.5, FPR 0.5 vs 0
    y = [0, 0, 0, 0, 1, 1] * 2
    approved = [1, 1, 1, 0, 1, 0] + [1, 1, 0, 0, 0, 0]
    groups = ['female'] * 6 + ['male'] * 6
    fl = lc.fairlearn_attribute(y, approved, groups, 'female')
    assert fl['equalized_odds_tpr_diff'] == pytest.approx(0.25)
    assert fl['equalized_odds_fpr_diff'] == pytest.approx(0.5)
    assert fl['equalized_odds_difference'] == pytest.approx(0.5)
    assert fl['di_ratio'] == pytest.approx((4 / 6) / (2 / 6))


def test_cross_check_flags_mismatch(caplog):
    y, p, gender, age = _sample(500)
    report = fm.fairness_report(y, p, gender=gender)
    report['by_attribute']['gender']['di_ratio'] += 0.01  # 일부러 틀린 값
    check = lc.cross_check(report, y, p, gender=gender)
    assert check['match'] is False
    assert check['by_attribute']['gender']['values']['di_ratio']['match'] is False
    assert any('교차 확인' in r.message for r in caplog.records)


def _split_sample(n, seed):
    y, p, gender, age = _sample(n, seed)
    return y, p, gender, age


@pytest.mark.parametrize('constraint', ['demographic_parity', 'equalized_odds'])
def test_threshold_optimizer_report_shape(constraint):
    fit = _split_sample(3000, 1)
    ev = _split_sample(3000, 2)
    r = lc.threshold_optimizer_report(*fit, *ev, constraints=constraint)
    assert r['method'] == f'fairlearn_threshold_optimizer_{constraint}'
    assert r['threshold_fit_on'] == 'validation'
    for side in ('before', 'after'):
        assert set(r[side]) >= {'di_ratio', 'equalized_odds_tpr_diff', 'equalized_odds_fpr_diff', 'by_attribute'}
        assert 0 <= r['overall'][side]['approval_rate'] <= 1


def test_threshold_optimizer_demographic_parity_evens_approval_rates():
    # 기준값을 정한 데이터에 그대로 적용하면 6개 집단 승인율이 거의 같아야 한다
    data = _split_sample(6000, 3)
    r = lc.threshold_optimizer_report(*data, *data, constraints='demographic_parity')
    before = r['before']['by_attribute']['age_group']['approval_rates'].values()
    after = r['after']['by_attribute']['age_group']['approval_rates'].values()
    assert max(after) - min(after) < max(before) - min(before)
    assert max(after) - min(after) < 0.05


def test_threshold_optimizer_is_reproducible():
    fit = _split_sample(2000, 4)
    ev = _split_sample(2000, 5)
    a = lc.threshold_optimizer_report(*fit, *ev, constraints='demographic_parity')
    b = lc.threshold_optimizer_report(*fit, *ev, constraints='demographic_parity')
    assert a['after']['di_ratio'] == b['after']['di_ratio']
