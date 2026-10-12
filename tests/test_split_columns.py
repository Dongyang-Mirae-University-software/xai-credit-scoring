import pandas as pd

from src.data.preprocessor import _to_split_csv_frame


def test_split_csv_uses_lowercase_names_and_string_age_group():
    df = pd.DataFrame({
        'customer_id': [7, 8, 9],
        'is_thin_filer': [1, 0, 0],
        'gender': [1, 0, 1],
        'age_group': [0, 1, 2],
    })
    out = _to_split_csv_frame(df)
    assert list(out.columns) == ['customer_id', 'is_thin_filer', 'gender', 'age_group']
    assert list(out['age_group']) == ['20-34', '35-54', '55+']
    assert list(df['age_group']) == [0, 1, 2]
