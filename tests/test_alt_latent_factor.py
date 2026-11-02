import numpy as np
import pandas as pd
import pytest

from src.data.simulator import (
    ALT_TARGET_CORR, DELINQUENCY_COLS, REVOLVING_COL,
    financial_risk_index, generate_alternative_data,
)

ALT = ['telecom_payment_rate', 'utility_payment_rate', 'spending_consistency',
       'regular_payment_count', 'app_login_frequency']


def _sample(n=20000, linked=True):
    """금융 칸이 있는 작은 표. linked=False면 금융 칸이 정답과 무관하다."""
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.1).astype(int)
    rate = 0.1 + 1.5 * y if linked else np.full(n, 0.3)
    df = pd.DataFrame({'SeriousDlqin2yrs': y, 'age': rng.integers(21, 90, n)})
    for col in DELINQUENCY_COLS:
        df[col] = rng.poisson(rate)
    df[REVOLVING_COL] = np.clip(rng.normal(0.3 + (0.3 * y if linked else 0), 0.2), 0, None)
    df['NumberOfOpenCreditLinesAndLoans'] = rng.integers(0, 10, n)
    df['NumberRealEstateLoansOrLines'] = rng.integers(0, 3, n)
    return df


@pytest.fixture(autouse=True)
def _no_save(monkeypatch):
    # generate_alternative_data는 결과를 data/intermediate에 저장하므로, 테스트에서는 저장을 막는다
    monkeypatch.setattr(pd.DataFrame, 'to_csv', lambda self, *args, **kwargs: None)


def test_each_variable_hits_target_correlation():
    out = generate_alternative_data(_sample(), random_seed=42)
    corr = out[ALT].corrwith(out['SeriousDlqin2yrs'])
    assert (corr < 0).all()
    assert np.allclose(corr.abs(), ALT_TARGET_CORR, atol=0.005)


def test_variables_share_common_factor():
    # 공통 성향 때문에 다섯 변수가 서로 강하게 함께 움직인다 (합쳐도 노이즈가 상쇄되지 않음)
    out = generate_alternative_data(_sample(), random_seed=42)
    pair = out[ALT].corr().values[np.triu_indices(len(ALT), k=1)]
    assert pair.min() > 0.5


def test_variables_are_linked_to_financial_risk():
    # 금융 위험이 클수록 대안 변수(성실할수록 큼)가 낮다
    df = _sample()
    out = generate_alternative_data(df, random_seed=42)
    risk = financial_risk_index(df)
    assert all(np.corrcoef(out[col], risk)[0, 1] < -0.1 for col in ALT)


def test_risk_index_does_not_use_target():
    df = _sample()
    flipped = df.assign(SeriousDlqin2yrs=1 - df['SeriousDlqin2yrs'])
    np.testing.assert_allclose(financial_risk_index(df), financial_risk_index(flipped))


def test_weak_latent_raises():
    # 금융 칸이 정답과 무관하고 정답 비중이 0이면 목표 상관을 만들 수 없다
    with pytest.raises(ValueError):
        generate_alternative_data(_sample(linked=False), random_seed=42, latent_weight=0.0)
