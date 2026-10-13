"""성능 비교 함수 테스트 (numpy·sklearn 없이 돌아감)"""
import math

import pytest

from src.explainer import performance as perf


def test_auc_perfect_and_reversed():
    """부도자에게 항상 높은 점수면 1.0, 거꾸로면 0.0"""
    y = [0, 0, 1, 1]
    assert perf.roc_auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert perf.roc_auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auc_ties_count_half():
    """점수가 모두 같으면 동전 던지기(0.5)"""
    assert perf.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auc_known_value():
    """sklearn.metrics.roc_auc_score([0,0,1,1],[0.1,0.4,0.35,0.8]) == 0.75 와 같은 값"""
    assert perf.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_ks_known_value():
    """같은 예시에서 TPR - FPR 의 최댓값은 0.5"""
    assert perf.ks_statistic([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.5)


def test_confusion_matrix_counts():
    cm = perf.confusion_matrix([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], threshold=0.5)
    assert (cm['tp'], cm['fp'], cm['tn'], cm['fn']) == (1, 1, 1, 1)
    assert cm['precision'] == 0.5 and cm['recall'] == 0.5 and cm['f1'] == 0.5


def test_psi_same_distribution_is_zero():
    scores = [i / 100 for i in range(100)]
    assert perf.psi(scores, scores) == pytest.approx(0.0, abs=1e-9)


def test_psi_shifted_distribution_is_large():
    base = [i / 100 for i in range(100)]
    shifted = [min(0.99, s + 0.5) for s in base]
    assert perf.psi(base, shifted) > 0.1


def test_thin_filer_uplift():
    """씬파일러만 놓고 대안 데이터 포함 모델이 더 잘 구분하면 향상이 양수"""
    y = [0, 1, 0, 1, 0, 1]
    without_alt = [0.5, 0.5, 0.5, 0.5, 0.2, 0.9]
    with_alt = [0.2, 0.8, 0.3, 0.7, 0.2, 0.9]
    thin = [1, 1, 1, 1, 0, 0]
    r = perf.thin_filer_uplift(y, without_alt, with_alt, thin)
    assert r['n_thin_filer'] == 4
    assert r['auc_without_alt'] == 0.5 and r['auc_with_alt'] == 1.0
    assert r['uplift'] == pytest.approx(0.5)


def test_thin_filer_uplift_needs_thin_filers():
    with pytest.raises(ValueError):
        perf.thin_filer_uplift([0, 1], [0.1, 0.9], [0.1, 0.9], [0, 0])


def test_evaluate_has_api_fields_and_disclaimer():
    """/metrics 응답 형식(auc, ks, precision, recall, f1)과 면책 조항"""
    r = perf.evaluate([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], threshold=0.5, train_scores=[0.1, 0.4, 0.35, 0.8])
    for key in ('auc', 'ks', 'precision', 'recall', 'f1', 'psi', 'roc_curve', 'pr_curve'):
        assert key in r
    assert r['disclaimer'] == perf.DISCLAIMER
    assert r['pass']['auc'] is (r['auc'] >= 0.78)
    assert not math.isnan(r['psi'])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        perf.roc_auc([0, 1], [0.5])
