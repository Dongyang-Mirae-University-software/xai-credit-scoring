import numpy as np
import pandas as pd

from src.data.simulator import generate_alternative_data

ALT = ['telecom_payment_rate', 'utility_payment_rate', 'spending_consistency',
       'regular_payment_count', 'app_login_frequency']


def test_alt_variables_are_lower_for_defaulters(monkeypatch):
    monkeypatch.setattr(pd.DataFrame, 'to_csv', lambda self, *args, **kwargs: None)
    rng = np.random.default_rng(0)
    n = 20000
    df = pd.DataFrame({
        'SeriousDlqin2yrs': (rng.random(n) < 0.2).astype(int),
        'age': rng.integers(21, 90, n),
    })
    np.random.seed(42)
    out = generate_alternative_data(df, thin_filer_ratio=None, bias_ratio=0, random_seed=42)

    means = out.groupby('SeriousDlqin2yrs')[ALT].mean()
    assert (means.loc[1] < means.loc[0]).all()
    assert means.loc[0, 'telecom_payment_rate'] >= 0.85

    corr = out[ALT + ['SeriousDlqin2yrs']].corr()['SeriousDlqin2yrs'][ALT]
    assert (corr < 0).all()
    assert (corr.abs() >= 0.3).all() and (corr.abs() <= 0.5).all()
