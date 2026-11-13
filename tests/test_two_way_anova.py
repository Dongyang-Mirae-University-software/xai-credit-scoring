"""보너스 3: Two-way ANOVA (src/statistics/two_way_anova.py) 테스트."""

import numpy as np
import pandas as pd
import pytest

from src.statistics import two_way_anova as tw


def _records(cell_means, n=6, noise=0.002, seed=0):
    """cell_means: {(segment, config): 평균}. 폴드 n개, 폴드마다 칸 4개."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        for (seg, cfg), m in cell_means.items():
            out.append({'fold_id': k, 'segment': seg, 'config': cfg, 'auc': m + rng.normal(0, noise)})
    return out


ADDITIVE = {('씬파일러', '전통'): 0.74, ('씬파일러', '통합'): 0.77, ('일반', '전통'): 0.86, ('일반', '통합'): 0.89}
INTERACT = {('씬파일러', '전통'): 0.74, ('씬파일러', '통합'): 0.81, ('일반', '전통'): 0.86, ('일반', '통합'): 0.89}


def _rows(result):
    return {r['source']: r for r in result['table']}


def test_sum_of_squares_add_up():
    r = tw.two_way_anova(_records(INTERACT))
    ss = sum(row['ss'] for row in r['table'])
    assert ss == pytest.approx(r['ss_total'])
    assert [row['df'] for row in r['table']] == [1, 1, 1, 4 * (6 - 1)]


def test_no_interaction_when_effects_add():
    rows = _rows(tw.two_way_anova(_records(ADDITIVE)))
    assert rows['segment']['is_significant'] and rows['config']['is_significant']
    assert not rows['segment×config']['is_significant']


def test_interaction_detected():
    r = tw.two_way_anova(_records(INTERACT))
    assert _rows(r)['segment×config']['is_significant']
    assert r['cell_means']['씬파일러|통합'] == pytest.approx(0.81, abs=0.005)


def test_matches_one_way_when_single_factor():
    """세그먼트 효과만 있을 때, 세그먼트 주효과 F 는 칸을 무시한 One-way ANOVA 와 같은 SS 를 쓴다."""
    from scipy import stats
    rec = _records({('씬파일러', '전통'): 0.75, ('씬파일러', '통합'): 0.75, ('일반', '전통'): 0.85, ('일반', '통합'): 0.85})
    r = _rows(tw.two_way_anova(rec))
    thin = [x['auc'] for x in rec if x['segment'] == '씬파일러']
    normal = [x['auc'] for x in rec if x['segment'] == '일반']
    grand = np.mean(thin + normal)
    ss_between = len(thin) * (np.mean(thin) - grand) ** 2 + len(normal) * (np.mean(normal) - grand) ** 2
    assert r['segment']['ss'] == pytest.approx(ss_between)
    assert stats.f_oneway(thin, normal).pvalue < 0.05


def test_unbalanced_rejected():
    rec = _records(ADDITIVE)
    with pytest.raises(ValueError):
        tw.two_way_anova(rec[:-1])


def test_paired_did_and_interpretation():
    rec = _records(INTERACT)
    paired = tw.paired_checks(rec)
    assert paired['simple_effects']['씬파일러']['mean'] == pytest.approx(0.07, abs=0.005)
    assert paired['simple_effects']['일반']['mean'] == pytest.approx(0.03, abs=0.005)
    assert paired['interaction_did']['mean'] == pytest.approx(0.04, abs=0.005)
    assert paired['interaction_did']['p_value'] < 0.05
    text = tw.interpret(tw.two_way_anova(rec), paired)
    assert any('씬파일러에서 더 크다' in s for s in text)


def test_fold_auc_records_shape():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    n = 600
    X = pd.DataFrame({'fin': rng.normal(size=n), 'alt': rng.normal(size=n)})
    y = (X['fin'] + X['alt'] + rng.normal(size=n) > 0.8).astype(int).to_numpy()
    thin = rng.random(n) < 0.3

    def select(X, data_type):
        return X[['fin']] if data_type == 'financial_only' else X[['fin', 'alt']]

    rec = tw.fold_auc_records(X, y, thin, {'전통': LogisticRegression(), '통합': LogisticRegression()}, select,
                              n_splits=3, n_repeats=2, seed=0)
    assert len(rec) == 3 * 2 * 4
    r = tw.two_way_anova(rec)
    assert r['n_per_cell'] == 6
    assert r['main_means']['config']['통합'] > r['main_means']['config']['전통']
