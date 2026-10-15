import numpy as np
import pandas as pd

from src.data.simulator import generate_alternative_data


def _sample(n=3000):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'SeriousDlqin2yrs': (rng.random(n) < 0.1).astype(int),
        'age': rng.integers(21, 90, n),
    })


def test_same_seed_gives_identical_result():
    a = generate_alternative_data(_sample(), thin_filer_ratio=None, bias_ratio=0, random_seed=42)
    b = generate_alternative_data(_sample(), thin_filer_ratio=None, bias_ratio=0, random_seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_result():
    a = generate_alternative_data(_sample(), thin_filer_ratio=None, bias_ratio=0, random_seed=42)
    b = generate_alternative_data(_sample(), thin_filer_ratio=None, bias_ratio=0, random_seed=7)
    assert not a['telecom_payment_rate'].equals(b['telecom_payment_rate'])
    assert not a['gender'].equals(b['gender'])
