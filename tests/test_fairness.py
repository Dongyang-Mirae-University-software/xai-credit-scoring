"""공정성 지표와 Bias Mitigation 테스트 (이슈 #39)"""
import json
import logging
from pathlib import Path

import pytest

from src.fairness import metrics as fm
from src.fairness import mitigation as mt

# 대시보드 mock 은 없어졌고(#48), 같은 형식의 API 응답 예시를 기준으로 비교한다
MOCK_DIR = Path(__file__).resolve().parent.parent / 'src' / 'api' / 'examples'


def test_normalize_gender_follows_column_spec():
    # docs/column_spec.md: 0 = 여성, 1 = 남성
    assert fm.normalize_gender([0, 1, 'F', 'male']) == ['female', 'male', 'female', 'male']
    with pytest.raises(ValueError):
        fm.normalize_gender([2])


def test_age_group_three_bins_from_age():
    assert fm.normalize_age_group(ages=[19, 34, 35, 54, 55, 80]) == ['20-34', '20-34', '35-54', '35-54', '55+', '55+']
    with pytest.raises(ValueError):
        fm.normalize_age_group(values=[0, 1, 4])  # 예전 5구간 숫자는 받지 않음


def test_disparate_impact_ratio():
    # 여성 10명 중 4명 승인, 남성 10명 중 8명 승인 → 0.4 / 0.8 = 0.5
    approved = [1] * 4 + [0] * 6 + [1] * 8 + [0] * 2
    groups = ['female'] * 10 + ['male'] * 10
    assert fm.disparate_impact(approved, groups, 'female') == pytest.approx(0.5)


@pytest.mark.parametrize('female_ok, warn', [(4, True), (8, False), (10, False), (16, True)])
def test_di_warning_outside_08_125(caplog, female_ok, warn):
    # 남성 승인율 0.4 기준. 여성 0.2(0.5배) / 0.4(1.0) / 0.5(1.25, 경계는 통과) / 0.8(2.0배)
    approved = [1] * female_ok + [0] * (20 - female_ok) + [1] * 8 + [0] * 12
    groups = ['female'] * 20 + ['male'] * 20
    with caplog.at_level(logging.WARNING, logger='src.fairness.metrics'):
        fm.disparate_impact(approved, groups, 'female', 'gender')
    assert any('DI Ratio' in r.message for r in caplog.records) is warn


def test_equalized_odds_tpr_fpr_diff():
    # 정상(y=0) 승인 비율 = TPR, 연체(y=1) 승인 비율 = FPR
    y = [0, 0, 0, 0, 1, 1] * 2
    approved = [1, 1, 1, 0, 1, 0] + [1, 1, 0, 0, 0, 0]
    groups = ['female'] * 6 + ['male'] * 6
    eo = fm.equalized_odds(y, approved, groups, 'female')
    assert eo['tpr_diff'] == pytest.approx(0.75 - 0.5)
    assert eo['fpr_diff'] == pytest.approx(0.5 - 0.0)
    assert eo['passed'] is False


def _biased_sample():
    # 남성의 부도 확률이 체계적으로 높게 나오는 예시 (승인 기준 0.5)
    y, p, g, a = [], [], [], []
    for i in range(200):
        male = i % 2 == 0
        y.append(1 if i % 10 == 0 else 0)
        p.append(min(0.95, 0.30 + (0.15 if male else 0.0) + (i % 7) * 0.02))
        g.append('male' if male else 'female')
        a.append('55+' if i % 3 == 0 else '35-54')
    return y, p, g, a


def test_fairness_report_has_dashboard_mock_keys():
    y, p, g, a = _biased_sample()
    report = fm.fairness_report(y, p, gender=g, age_group=a)
    mock = json.loads((MOCK_DIR / 'fairness.json').read_text(encoding='utf-8'))
    assert set(mock) <= set(report)
    assert set(report['by_attribute']) == {'gender', 'age_group'}
    assert report['by_attribute']['gender']['protected_group'] == 'female'


def test_reweighing_balances_default_rate_across_groups():
    y = [1, 0, 0, 0] * 5 + [1, 1, 0, 0] * 5   # A 연체율 25%, B 50%
    g = ['A'] * 20 + ['B'] * 20
    w = mt.reweighing_weights(y, g)
    assert sum(w) / len(w) == pytest.approx(1.0)

    def weighted_rate(group):
        num = sum(wi for wi, yi, gi in zip(w, y, g) if gi == group and yi == 1)
        den = sum(wi for wi, gi in zip(w, g) if gi == group)
        return num / den
    assert weighted_rate('A') == pytest.approx(weighted_rate('B'))


def test_threshold_adjustment_brings_di_into_range():
    y, p, g, a = _biased_sample()
    report = mt.mitigation_report(y, p, gender=g, age_group=a, method='threshold_adjustment')
    before = report['before']['by_attribute']['gender']['di_ratio']
    after = report['after']['by_attribute']['gender']['di_ratio']
    assert not (fm.DI_MIN <= before <= fm.DI_MAX)
    assert fm.DI_MIN <= after <= fm.DI_MAX
    mock = json.loads((MOCK_DIR / 'fairness_mitigation.json').read_text(encoding='utf-8'))
    for side in ('before', 'after'):
        assert set(mock[side]) <= set(report[side])
    assert '| DI Ratio |' in mt.comparison_table(report)


def test_reweighting_report_needs_retrained_scores():
    y, p, g, a = _biased_sample()
    with pytest.raises(ValueError):
        mt.mitigation_report(y, p, gender=g, method='reweighting')
    report = mt.mitigation_report(y, p, gender=g, method='reweighting', p_after=[0.2] * len(p))
    assert report['after']['di_ratio'] == pytest.approx(1.0)


def test_threshold_adjustment_fits_on_validation_and_applies_to_test():
    # Validation과 Test의 점수 분포가 다르면, 기준값은 Validation 기준이어야 한다
    y_v, p_v, g_v, a_v = _biased_sample()
    y_t, p_t, g_t, a_t = _biased_sample()
    p_t = [min(0.99, x + 0.05) for x in p_t]          # Test 점수를 일부러 조금 높임
    fitted, _ = mt.group_thresholds(p_v, mt.combine_groups(g_v, a_v))
    report = mt.mitigation_report(y_t, p_t, gender=g_t, age_group=a_t, method='threshold_adjustment',
                                  fit_p_default=p_v, fit_gender=g_v, fit_age_group=a_v)
    assert report['threshold_fit_on'] == 'validation'
    assert report['group_thresholds'] == fitted
    test_fitted, _ = mt.group_thresholds(p_t, mt.combine_groups(g_t, a_t))
    assert report['group_thresholds'] != test_fitted


def test_threshold_adjustment_without_fit_data_warns(caplog):
    y, p, g, a = _biased_sample()
    with caplog.at_level(logging.WARNING, logger='src.fairness.mitigation'):
        report = mt.mitigation_report(y, p, gender=g, age_group=a, method='threshold_adjustment')
    assert report['threshold_fit_on'] == 'evaluation_data'
    assert any('Validation' in r.message for r in caplog.records)
