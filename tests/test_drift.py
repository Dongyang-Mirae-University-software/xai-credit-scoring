"""보너스 4: PSI + CSI 드리프트 탐지 (src/statistics/drift.py) 테스트."""

import logging
import math

import numpy as np
import pandas as pd
import pytest

from src.statistics import drift


def test_same_distribution_is_stable():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20000)
    b = rng.normal(size=20000)
    t = drift.stability_table(a, b)
    assert t['index'] < 0.01 and t['grade'] == '안정'
    assert sum(r['expected'] for r in t['bins']) == pytest.approx(1.0)


def test_matches_hand_calculation_discrete():
    exp = [0] * 50 + [1] * 50
    act = [0] * 20 + [1] * 80
    t = drift.stability_table(exp, act)
    hand = (0.2 - 0.5) * math.log(0.2 / 0.5) + (0.8 - 0.5) * math.log(0.8 / 0.5)
    assert t['binning'] == 'discrete'
    assert t['index'] == pytest.approx(hand, rel=1e-4)


def test_out_of_range_values_get_own_bin():
    exp = [18, 19, 20, 21] * 25
    act = [8, 9, 10, 11] * 25          # 모두 기준 최솟값보다 작다
    t = drift.stability_table(exp, act)
    first = t['bins'][0]
    assert '기준 범위 밖' in first['label'] and first['actual'] == pytest.approx(1.0)
    assert t['grade'] == '경고'


def test_grade_thresholds():
    assert drift.grade(0.05) == '안정'
    assert drift.grade(0.1) == '주의'
    assert drift.grade(0.2499) == '주의'
    assert drift.grade(0.25) == '경고'


def test_shift_detected_and_score_psi_matches_existing():
    from src.explainer import performance as perf
    rng = np.random.default_rng(1)
    a = rng.normal(size=10000)
    b = rng.normal(0.5, 1, size=10000)
    assert drift.psi(a, b) >= drift.STABLE
    # 연속 값에서는 기존 performance.psi 와 같은 구간 방식이라 거의 같은 값이 나와야 한다
    assert drift.psi(a, b) == pytest.approx(perf.psi(a, b), rel=0.02)


def test_drift_report_flags_only_changed_column(caplog):
    rng = np.random.default_rng(2)
    n = 5000
    base = pd.DataFrame({'x': rng.normal(size=n), 'count': rng.poisson(3, size=n)})
    new = base.copy()
    new['count'] = new['count'] + 3
    with caplog.at_level(logging.WARNING, logger='src.statistics.drift'):
        rep = drift.drift_report(base, new, ['x', 'count'], rng.random(n), rng.random(n), name='시험')
    assert rep['flagged'] == ['count']
    assert rep['csi']['x']['grade'] == '안정'
    assert 'count' in rep['top_bins']
    assert any('CSI count' in m for m in caplog.messages)
