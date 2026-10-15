"""ANOVA 통계 검증 테스트 (이슈 #42)"""
import json
from pathlib import Path

import pytest

from src.statistics import anova

MOCK = Path(__file__).resolve().parent.parent / 'src' / 'dashboard' / 'mock' / 'statistical_test' / 'anova.json'


def test_eta_squared_interpretation_follows_original_table():
    assert anova.interpret_eta_squared(0.005) == 'Negligible'
    assert anova.interpret_eta_squared(0.01) == 'Small'
    assert anova.interpret_eta_squared(0.06) == 'Medium'
    assert anova.interpret_eta_squared(0.14) == 'Large'


def test_eta_squared_value():
    # 집단 평균 1, 3 / 전체 평균 2 → 집단 간 제곱합 4, 전체 제곱합 4 + 집단 안 0 = 4 → η² = 1
    assert anova.eta_squared([[1, 1], [3, 3]]) == pytest.approx(1.0)
    assert anova.eta_squared([[1, 3], [1, 3]]) == pytest.approx(0.0)


def test_one_way_anova_matches_scipy_and_runs_tukey_when_significant():
    from scipy import stats
    groups = {'전통모델': [0.70, 0.71, 0.72, 0.71, 0.70],
              '대안모델': [0.72, 0.73, 0.72, 0.74, 0.73],
              '통합모델': [0.78, 0.79, 0.80, 0.79, 0.78]}
    r = anova.one_way_anova(groups, 'test')
    f, p = stats.f_oneway(*groups.values())
    assert r['f_statistic'] == pytest.approx(f)
    assert r['p_value'] == pytest.approx(p)
    assert r['is_significant'] is True
    assert [t['comparison'] for t in r['post_hoc_tukey']] == ['전통 vs 대안', '전통 vs 통합', '대안 vs 통합']
    assert r['post_hoc_tukey'][1]['mean_diff'] == pytest.approx(0.708 - 0.788)


def test_no_tukey_when_not_significant():
    r = anova.one_way_anova({'A': [0.70, 0.75, 0.72], 'B': [0.71, 0.74, 0.73]})
    assert r['is_significant'] is False
    assert r['post_hoc_tukey'] == []


def test_needs_two_groups_with_two_scores():
    with pytest.raises(ValueError):
        anova.one_way_anova({'A': [0.7, 0.8]})
    with pytest.raises(ValueError):
        anova.one_way_anova({'A': [0.7, 0.8], 'B': [0.7]})


def _run(alg, method, dt, base, start, status='FINISHED', folds=5):
    metrics = {f'fold_{k}_roc_auc': base + 0.001 * k for k in range(1, folds + 1)}
    return {'status': status, 'start_time': start,
            'params': {'algorithm': alg, 'method': method, 'data_type': dt}, 'metrics': metrics}


def _records():
    rs = []
    t = 1000
    for alg, a_base in (('Logistic Regression', 0.0), ('XGBoost', 0.03), ('LightGBM', 0.031)):
        for method in ('m1', 'm2'):
            for dt, d_base in (('financial_only', 0.64), ('alternative_only', 0.72), ('integrated', 0.75)):
                t += 1
                rs.append(_run(alg, method, dt, d_base + a_base, t))
    rs.append(_run('XGBoost', 'm1', 'integrated', 0.10, 1, ))          # 예전 기록 → 최신으로 대체돼야 함
    rs.append(_run('XGBoost', 'm1', 'integrated', 0.10, 99999, status='RUNNING'))  # 끝나지 않은 run 제외
    rs.append(_run('LightGBM', 'm9', 'integrated', 0.10, 99999, folds=3))         # 폴드 부족 제외
    return rs


def test_group_scores_uses_latest_finished_runs_only():
    g = anova.group_scores(_records(), 'data_type', anova.DATA_TYPE_LABELS)
    assert list(g) == ['전통모델', '대안모델', '통합모델']
    assert len(g['통합모델']) == 6 * 5           # 알고리즘 3 × 방법 2 × 폴드 5
    assert min(g['통합모델']) > 0.5               # 0.10 짜리 옛 기록이 섞이지 않음
    g2 = anova.group_scores(_records(), 'algorithm', anova.ALGORITHM_LABELS, where={'data_type': 'integrated'})
    assert list(g2) == ['LogisticRegression', 'XGBoost', 'LightGBM']
    assert all(len(v) == 10 for v in g2.values())


def test_report_has_dashboard_mock_keys_and_interpretation():
    report = anova.anova_report(_records())
    mock = json.loads(MOCK.read_text(encoding='utf-8'))
    for key in ('test_1', 'test_2'):
        assert set(mock[key]) <= set(report[key])
        for row in report[key]['post_hoc_tukey']:
            assert set(mock[key]['post_hoc_tukey'][0]) <= set(row)
        assert any('F =' in s for s in report['interpretation'][key])
    assert report['test_2']['groups'] == ['LogisticRegression', 'XGBoost', 'LightGBM']
    assert '교육 목적' in report['disclaimer']


def test_skeleton_function_names_still_work():
    # 초기 구조(8df9d2b)의 함수 이름으로도 부를 수 있어야 함
    assert anova.run_anova_feature_group(_records())['groups'] == ['전통모델', '대안모델', '통합모델']
    assert anova.run_anova_algorithm(_records())['groups'] == ['LogisticRegression', 'XGBoost', 'LightGBM']


def test_korean_particles_in_interpretation():
    assert anova._josa('통합', '이', '가') == '통합이'
    assert anova._josa('XGBoost', '이', '가') == 'XGBoost가'
    assert anova._josa('LightGBM', '과', '와') == 'LightGBM과'


def test_records_from_rest():
    payload = {'runs': [{'info': {'run_name': 'r', 'status': 'FINISHED', 'start_time': '5'},
                         'data': {'params': [{'key': 'algorithm', 'value': 'XGBoost'}],
                                  'metrics': [{'key': 'fold_1_roc_auc', 'value': 0.7}]}}]}
    rec = anova.records_from_rest(payload)[0]
    assert rec['params']['algorithm'] == 'XGBoost' and rec['metrics']['fold_1_roc_auc'] == 0.7
    assert rec['start_time'] == 5
