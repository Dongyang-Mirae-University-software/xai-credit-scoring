import pandas as pd

from src.data.simulator import add_demographic_features


def test_age_group_has_three_bands():
    df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 0, 0, 0, 0, 0, 0, 0],
        'age': [21, 34, 35, 54, 55, 80, 101, 109],
    })
    result = add_demographic_features(df, bias_ratio=0)
    assert list(result['age_group']) == [0, 0, 1, 1, 2, 2, 2, 2]
