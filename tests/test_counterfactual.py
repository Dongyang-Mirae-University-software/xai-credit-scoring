"""보너스 2: Counterfactual Explanation 테스트 (가짜 모델로 탐색 규칙 확인 + 실제 API)."""
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import model
from src.api.main import app
from src.api.schemas import CUSTOMER_EXAMPLE
from src.explainer import counterfactual as cf

BASE = {
    'RevolvingUtilizationOfUnsecuredLines': 0.9, 'age': 30, 'NumberOfTime30-59DaysPastDueNotWorse': 2,
    'DebtRatio': 0.6, 'MonthlyIncome': 3000.0, 'NumberOfOpenCreditLinesAndLoans': 5,
    'NumberOfTimes90DaysLate': 1, 'NumberRealEstateLoansOrLines': 0, 'NumberOfTime60-89DaysPastDueNotWorse': 0,
    'NumberOfDependents': 1, 'telecom_payment_rate': 0.8, 'utility_payment_rate': 0.75,
    'spending_consistency': 85.0, 'regular_payment_count': 10.0, 'app_login_frequency': 18.0,
}


def fake_model(frame):
    """한도 사용 비율이 높고 통신비 납부율이 낮을수록, 연체가 많을수록 위험한 가짜 모델."""
    z = (4 * frame['RevolvingUtilizationOfUnsecuredLines'] - 10 * (frame['telecom_payment_rate'] - 0.8)
         + 0.5 * frame['NumberOfTimes90DaysLate'] - 3.2)
    return 1 / (1 + np.exp(-z.values))


def test_levels_move_only_in_good_direction_within_train_range():
    down = cf.levels('RevolvingUtilizationOfUnsecuredLines', 0.9)
    assert down == sorted(down, reverse=True) and min(down) >= 0.0 and max(down) < 0.9
    up = cf.levels('telecom_payment_rate', 0.8)
    assert up == sorted(up) and max(up) <= 1.0
    assert cf.levels('telecom_payment_rate', 1.0) == []          # 이미 한계
    assert all(float(v).is_integer() for v in cf.levels('regular_payment_count', 10.0))
    assert max(cf.levels('MonthlyIncome', 3000.0)) == pytest.approx(4500.0)   # 월소득은 +50% 까지만


def test_single_change_finds_smallest_approving_value():
    single = cf.single_changes(BASE, fake_model, 0.5)
    r = single['RevolvingUtilizationOfUnsecuredLines']
    assert r is not None and r['new_probability'] < 0.5
    grid = cf.levels('RevolvingUtilizationOfUnsecuredLines', 0.9)
    prev = grid[grid.index(r['suggested']) - 1] if grid.index(r['suggested']) > 0 else 0.9
    assert fake_model(cf._batch(BASE, [{'RevolvingUtilizationOfUnsecuredLines': prev}]))[0] >= 0.5
    assert single['DebtRatio'] is None                              # 가짜 모델은 부채 비율을 보지 않음


def test_combined_changes_reach_approval_without_touching_immutable():
    comb = cf.combined_changes(BASE, fake_model, 0.5)
    assert comb['success'] and comb['new_probability'] < 0.5
    assert set(comb['changes']) <= set(cf.ACTIONABLE)
    assert not set(comb['changes']) & set(cf.IMMUTABLE)
    assert set(comb['changes']) <= {'RevolvingUtilizationOfUnsecuredLines', 'telecom_payment_rate'}  # 쓸모없는 칸은 되돌림


def test_combined_reports_failure_when_impossible():
    always_bad = lambda frame: np.full(len(frame), 0.99)  # noqa: E731
    comb = cf.combined_changes(BASE, always_bad, 0.5)
    assert comb['success'] is False and math.isclose(comb['new_probability'], 0.99)


def test_describe_uses_display_units():
    text = cf.describe('RevolvingUtilizationOfUnsecuredLines', 0.9, 0.4, cf.DISPLAY_UNITS)
    assert '90.0%' in text and '40.0%' in text and '낮추기' in text
    assert '4,500,000원' in cf.describe('MonthlyIncome', 3000, 4500, cf.DISPLAY_UNITS)


@pytest.mark.skipif(not model.ready(), reason='models/ 에 최종 모델 파일이 없음')
def test_counterfactual_api():
    client = TestClient(app)
    risky = {**CUSTOMER_EXAMPLE, 'RevolvingUtilizationOfUnsecuredLines': 0.95, 'NumberOfTimes90DaysLate': 2,
             'telecom_payment_rate': 80, 'utility_payment_rate': 72, 'spending_consistency': 84,
             'regular_payment_count': 10, 'app_login_frequency': 18}
    body = client.post('/counterfactual', json=risky).json()
    assert body['approval_status'] == 'REJECTED' and body['already_approved'] is False
    plan = body['combined']
    if plan['success']:
        assert plan['new_probability'] < body['approval_threshold']
    assert '나이' in body['immutable_features']
    safe = client.post('/counterfactual', json=CUSTOMER_EXAMPLE).json()
    if safe['already_approved']:
        assert safe['combined'] is None and safe['single_changes'] == []
