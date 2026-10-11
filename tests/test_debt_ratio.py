import numpy as np
import pandas as pd

from src.data.preprocessor import preprocess


def _original_debt_ratio(processed, scaler):
    i = list(scaler.feature_names_in_).index('DebtRatio')
    return processed['DebtRatio'] * scaler.scale_[i] + scaler.mean_[i]


def test_missing_income_debt_ratio_converted_to_ratio():
    df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 1, 0, 0],
        'DebtRatio': [0.3, 1000.0, 0.5, 2000.0],
        'MonthlyIncome': [5000.0, np.nan, 4000.0, np.nan],
        'age': [30, 40, 50, 60],
    })
    processed, scaler = preprocess(df)
    recovered = _original_debt_ratio(processed, scaler)
    assert len(processed) == 4
    assert np.allclose(recovered.values, [0.3, 1000 / 4500, 0.5, 2000 / 4500])


def test_income_present_rows_are_not_changed():
    df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 1],
        'DebtRatio': [0.3, 2.5],
        'MonthlyIncome': [5000.0, 3000.0],
        'age': [30, 40],
    })
    processed, scaler = preprocess(df)
    assert np.allclose(_original_debt_ratio(processed, scaler).values, [0.3, 2.5])
