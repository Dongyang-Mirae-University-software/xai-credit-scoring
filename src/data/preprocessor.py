"""
결측치 처리, 스케일링, 인코딩, Train/Validation/Test 분할.
분할은 Stratified Split만 사용한다.
"""


def preprocess(df):
    raise NotImplementedError


def split_data(df, target_col: str, train_ratio: float = 0.7,
                val_ratio: float = 0.15, test_ratio: float = 0.15,
                random_seed: int = 42):
    raise NotImplementedError
