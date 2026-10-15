import numpy as np
import pandas as pd

from src.data.simulator import add_demographic_features


def _base(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'age': rng.integers(21, 90, n),
        'app_login_frequency': rng.integers(5, 26, n),
    })


def test_bias_zero_changes_nothing():
    """bias_ratio=0이면 기본 데이터와 완전히 같다 (#41)"""
    a = add_demographic_features(_base(), 0.0, rng=np.random.default_rng(1))
    b = add_demographic_features(_base(), 0.0, rng=np.random.default_rng(1))
    pd.testing.assert_frame_equal(a, b)
    base = _base()
    assert (a['app_login_frequency'].values == base['app_login_frequency'].values).all()


def test_bias_raises_male_login_frequency_only():
    """bias_ratio>0이면 남성의 앱 로그인 빈도만 올라간다 (대리 변수)"""
    base = _base()
    out = add_demographic_features(base.copy(), 0.2, rng=np.random.default_rng(1))
    shift = 6  # round(30 * 0.2)
    male = out['gender'] == 1
    female = ~male
    before = base['app_login_frequency'].values
    after = out['app_login_frequency'].values
    assert (after[female.values] == before[female.values]).all()
    assert (after[male.values] >= before[male.values]).all()
    assert (after[male.values] - before[male.values] <= shift).all()
    assert after.min() >= 0 and after.max() <= 30


def test_bias_proxy_does_not_use_target():
    """정답 칸이 없으면 대리 변수 외의 값은 바뀌지 않는다 (정답과 무관하게 추가됨)"""
    out = add_demographic_features(_base(), 0.1, rng=np.random.default_rng(3))
    assert 'SeriousDlqin2yrs' not in out.columns
    assert out['gender'].isin([0, 1]).all()
