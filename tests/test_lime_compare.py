"""보너스 1: LIME vs SHAP 상위 5개 일치율 계산 테스트."""
import numpy as np
import pandas as pd
import pytest

from src.explainer import lime_compare as lc

COLS = ['a', 'b', 'c', 'd', 'e', 'f', 'g']


def test_top_k_features_uses_absolute_value():
    contrib = [0.1, -0.9, 0.3, 0.0, -0.5, 0.2, 0.05]
    assert lc.top_k_features(contrib, COLS, k=3) == ['b', 'e', 'c']


def test_top_k_agreement_ignores_order():
    assert lc.top_k_agreement(['a', 'b', 'c', 'd', 'e'], ['e', 'd', 'c', 'b', 'a']) == 1.0
    assert lc.top_k_agreement(['a', 'b', 'c', 'd', 'e'], ['a', 'b', 'x', 'y', 'z']) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        lc.top_k_agreement(['a'], ['a'], k=0)


def test_sign_agreement_counts_same_direction():
    shap_row = np.array([0.5, -0.2, 0.1, 0, 0, 0, 0])
    lime_row = np.array([0.3, 0.4, 0.2, 0, 0, 0, 0])
    assert lc.sign_agreement(shap_row, lime_row, ['a', 'b', 'c'], COLS) == pytest.approx(2 / 3)
    assert lc.sign_agreement(shap_row, lime_row, [], COLS) is None


def test_compare_and_summarize():
    shap_vals = np.array([[5, 4, 3, 2, 1, 0, 0], [5, 4, 3, 2, 1, 0, 0]], dtype=float)
    lime_vals = np.array([[5, 4, 3, 2, 1, 0, 0], [0, 0, 3, 2, 1, 5, 4]], dtype=float)
    rows = lc.compare(shap_vals, lime_vals, COLS)
    assert [r['agreement'] for r in rows] == [1.0, pytest.approx(0.6)]
    s = lc.summarize(rows)
    assert s['n'] == 2
    assert s['mean_agreement'] == pytest.approx(0.8)
    assert s['overlap_count_distribution']['5'] == 1 and s['overlap_count_distribution']['3'] == 1
    assert s['top1_same_rate'] == 0.5
    freq = lc.top_k_frequency(rows, 'lime_top', COLS)
    assert freq['a'] == 0.5 and freq['c'] == 1.0


def test_lime_finds_the_only_informative_feature():
    """정답을 정하는 변수가 하나뿐이면 LIME 상위 1위도 그 변수여야 한다."""
    pytest.importorskip('lime')
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 4)), columns=['x0', 'x1', 'x2', 'x3'])
    y = (X['x2'] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    exp = lc.make_lime_explainer(X, random_state=0)
    w = lc.lime_contributions(exp, lambda a: model.predict_proba(pd.DataFrame(a, columns=X.columns)),
                              X.iloc[0].values, n_features=4, num_samples=1000)
    assert lc.top_k_features(w, list(X.columns), k=1) == ['x2']
