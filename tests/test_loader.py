import pandas as pd

from src.data.loader import load_gmsc


def test_load_gmsc_reads_training_file_only(tmp_path):
    # cs-training.csv 하나만 있어도 읽히고, 데이터프레임 하나를 돌려준다 (cs-test.csv는 쓰지 않음, 이슈 #101)
    path = tmp_path / 'cs-training.csv'
    pd.DataFrame({'SeriousDlqin2yrs': [0, 1], 'age': [30, 45]}).to_csv(path, index=False)
    df = load_gmsc(str(path))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['SeriousDlqin2yrs', 'age']
    assert len(df) == 2
