"""거절 사유 Top 5 테스트 (shap 없이 돌아감)"""
from src.explainer.reasons import age_group_of, build_rejection_reasons, reference_values


def test_age_group_matches_assignment():
    """원본 보호 속성 연령대 20-34 / 35-54 / 55+"""
    assert age_group_of(24) == '20-34'
    assert age_group_of(34) == '20-34'
    assert age_group_of(35) == '35-54'
    assert age_group_of(54) == '35-54'
    assert age_group_of(55) == '55+'


def test_reference_values_by_age_group():
    rows = [
        {'age': 25, 'telecom_payment_rate': 0.9},
        {'age': 30, 'telecom_payment_rate': 0.94},
        {'age': 60, 'telecom_payment_rate': 0.5},
    ]
    refs = reference_values(rows, ['telecom_payment_rate'])
    assert round(refs['20-34']['telecom_payment_rate'], 2) == 0.92
    assert refs['55+']['telecom_payment_rate'] == 0.5
    assert round(refs['_all']['telecom_payment_rate'], 4) == round((0.9 + 0.94 + 0.5) / 3, 4)


def test_reason_has_required_fields_and_sentence():
    """원본 형식: feature, current_value, shap_contribution, explanation + 예시 문장 형태"""
    refs = {'20-34': {'telecom_payment_rate': 0.92}, '_all': {'telecom_payment_rate': 0.9}}
    reasons = build_rejection_reasons(
        {'telecom_payment_rate': 0.78, 'age': 24},
        {'telecom_payment_rate': 0.15},
        references=refs, age=24)
    r = reasons[0]
    for key in ('feature', 'current_value', 'shap_contribution', 'explanation'):
        assert key in r
    assert r['shap_contribution'] == -0.15
    assert r['explanation'] == '통신비 정상납부율이 78%로, 동일 연령대 평균(92%) 대비 낮습니다 (기여도: -0.15)'


def test_korean_particles():
    """조사: 받침에 따라 이/가, 으로/로"""
    reasons = build_rejection_reasons(
        {'MonthlyIncome': 5400, 'NumberOfTimes90DaysLate': 3},
        {'MonthlyIncome': 0.2, 'NumberOfTimes90DaysLate': 0.3},
        references={'_all': {'MonthlyIncome': 6418, 'NumberOfTimes90DaysLate': 0.27}})
    texts = {r['feature']: r['explanation'] for r in reasons}
    assert texts['MonthlyIncome'].startswith('월소득이 5,400으로, 전체 평균(6,418) 대비 낮습니다')
    assert texts['NumberOfTimes90DaysLate'].startswith('90일 이상 연체 횟수가 3회로, 전체 평균(0.3회) 대비 높습니다')


def test_top5_sorted_and_only_unfavorable():
    shap = {f'f{i}': v for i, v in enumerate([0.5, -0.2, 0.1, 0.3, 0.05, 0.2, 0.01])}
    values = {k: 1 for k in shap}
    reasons = build_rejection_reasons(values, shap, top_k=5)
    assert [r['feature'] for r in reasons] == ['f0', 'f3', 'f5', 'f2', 'f4']
    assert all(r['shap_contribution'] < 0 for r in reasons)
    assert [r['rank'] for r in reasons] == [1, 2, 3, 4, 5]
