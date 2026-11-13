"""보너스 5: A/B 테스트 시뮬레이션 (src/statistics/ab_test.py) 테스트."""

import numpy as np
import pytest

from src.statistics import ab_test as ab


def _data(seed=0, n=20000):
    """B 가 A 보다 부도를 더 잘 가려내는 가짜 점수."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.07).astype(int)
    p_a = np.clip(0.3 + 0.25 * y + rng.normal(0, 0.15, n), 0, 1)
    p_b = np.clip(0.3 + 0.35 * y + rng.normal(0, 0.15, n), 0, 1)
    return y, p_a, p_b


def test_approval_metrics_counts():
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.3])
    m = ab.approval_metrics(y, p, 0.5)
    assert m['n_approved'] == 3 and m['approval_rate'] == pytest.approx(0.6)
    assert m['n_bad_approved'] == 1 and m['bad_rate'] == pytest.approx(1 / 3)
    assert m['bad_caught'] == 1


def test_threshold_for_approval_rate():
    p = np.linspace(0, 1, 1001)
    t = ab.threshold_for_approval_rate(p, 0.3)
    assert (p < t).mean() == pytest.approx(0.3, abs=0.002)


def test_threshold_for_bad_rate_respects_target():
    y, _, p_b = _data()
    t = ab.threshold_for_bad_rate(y, p_b, 0.02)
    m = ab.approval_metrics(y, p_b, t)
    assert m['bad_rate'] <= 0.02 + 1e-12
    # 조금만 더 승인하면 목표를 넘는다 (가장 많이 승인하는 기준값)
    assert ab.approval_metrics(y, p_b, np.nextafter(np.sort(p_b[p_b >= t])[0], 2))['bad_rate'] > 0.02 - 1e-3


def test_better_model_wins_all_policies():
    y, p_a, p_b = _data()
    r = ab.policy_comparison(y, p_a, p_b, 0.5, segment=np.arange(len(y)) % 5 == 0)
    same_app = r['policies']['same_approval_rate']
    same_bad = r['policies']['same_bad_rate']
    assert same_app['b']['approval_rate'] == pytest.approx(r['a']['approval_rate'], abs=0.001)
    assert same_app['bad_rate_change'] < 0
    assert same_bad['b']['bad_rate'] <= r['a']['bad_rate'] + 1e-12
    assert same_bad['approval_change'] > 0
    assert set(same_app['segments']) == {'씬파일러', '일반'}


def test_two_proportion_test_known_values():
    t = ab.two_proportion_test(50, 1000, 30, 1000)
    assert t['diff'] == pytest.approx(-0.02)
    assert t['p_value'] < 0.05 and t['ci_high'] < 0
    same = ab.two_proportion_test(50, 1000, 50, 1000)
    assert same['p_value'] == pytest.approx(1.0)


def test_repeated_ab_and_sample_size():
    y, p_a, p_b = _data(n=8000)
    rep = ab.repeated_ab(y, p_a, p_b, 0.5, 0.5, n_rep=50, seed=1)
    assert rep['n_rep'] == 50 and len(rep['bad_rate_diffs']) == 50
    assert 0.0 <= rep['share_significant_bad_rate'] <= 1.0
    n = ab.required_sample_size(0.0207, 0.0172, 0.768)
    assert 20000 < n['approved_per_arm'] < 30000
    assert n['applicants_total'] == 2 * n['applicants_per_arm']
