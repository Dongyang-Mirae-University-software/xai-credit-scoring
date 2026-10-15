"""전처리는 학습(Train) 행으로만 중앙값·스케일러를 계산해야 한다 (전처리 순서 수정)"""
import numpy as np
import pandas as pd

from src.data.preprocessor import preprocess, get_train_index


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'SeriousDlqin2yrs': (rng.random(n) < 0.1).astype(int),
        'age': rng.integers(21, 90, n).astype(float),
        'RevolvingUtilizationOfUnsecuredLines': rng.random(n),
        'NumberOfTime30-59DaysPastDueNotWorse': rng.integers(0, 3, n).astype(float),
        'MonthlyIncome': rng.integers(1000, 9000, n).astype(float),
        'DebtRatio': rng.random(n),
    })
    df.loc[df.index[:20], 'MonthlyIncome'] = np.nan
    df.loc[df.index[20:30], 'age'] = np.nan
    return df


def test_train_index_is_stratified_and_seeded():
    df = _frame()
    a = get_train_index(df, test_size=0.3, random_seed=42)
    b = get_train_index(df, test_size=0.3, random_seed=42)
    assert list(a) == list(b)
    assert abs(len(a) / len(df) - 0.7) < 0.01


def test_scaler_and_medians_use_only_fit_rows():
    """fit 행 밖의 극단값이 평균·중앙값을 바꾸면 안 된다"""
    df = _frame()
    fit_idx = get_train_index(df, test_size=0.3, random_seed=42)
    outside = df.index.difference(fit_idx)
    df.loc[outside, 'RevolvingUtilizationOfUnsecuredLines'] = 1e6  # fit 밖에만 극단값
    out, scaler, stats = preprocess(df.copy(), fit_index=fit_idx, return_stats=True)

    fit_values = df.loc[fit_idx, 'RevolvingUtilizationOfUnsecuredLines']
    assert abs(scaler.mean_[list(scaler.feature_names_in_).index('RevolvingUtilizationOfUnsecuredLines')]
               - fit_values.mean()) < 1e-9
    assert abs(stats['MonthlyIncome_median'] - df.loc[fit_idx, 'MonthlyIncome'].median()) < 1e-9
    assert stats['age_median'] == df.loc[fit_idx, 'age'].median()


def test_fit_index_none_keeps_old_behavior_for_scripts():
    """fit_index를 주지 않으면 예전처럼 전체 행으로 계산한다 (기존 스크립트 호환)"""
    df = _frame()
    out, scaler = preprocess(df.copy())
    assert out.shape[0] == len(df)
