"""씬파일러 판정 기준 테스트 (이슈 #15)"""
import pandas as pd
import pytest

from src.data.simulator import label_thin_filer


def _row(open_lines=5, real_estate=1, revolving=0.5, debt_ratio=0.3,
         d30=0, d60=0, d90=0, target=0):
    """기본값은 '일반 고객' 한 명. 인자를 바꿔 조건을 하나씩 검증한다."""
    return {
        'NumberOfOpenCreditLinesAndLoans': open_lines,
        'NumberRealEstateLoansOrLines': real_estate,
        'RevolvingUtilizationOfUnsecuredLines': revolving,
        'DebtRatio': debt_ratio,
        'NumberOfTime30-59DaysPastDueNotWorse': d30,
        'NumberOfTime60-89DaysPastDueNotWorse': d60,
        'NumberOfTimes90DaysLate': d90,
        'SeriousDlqin2yrs': target,
    }


def _thin_filer_row(**overrides):
    """기준 2를 모두 만족하는 한 명"""
    row = _row(open_lines=1, real_estate=0, revolving=0.2, debt_ratio=0.3)
    row.update(overrides)
    return row


def _judge(rows):
    return label_thin_filer(pd.DataFrame(rows))['is_thin_filer']


def test_all_conditions_met_is_thin_filer():
    """기준 2의 모든 조건을 만족하면 씬파일러"""
    assert _judge([_thin_filer_row()]).iloc[0] == 1


def test_many_accounts_is_not_thin_filer():
    """보유 대출·신용 한도가 2개 이상이면 제외"""
    assert _judge([_thin_filer_row(**{'NumberOfOpenCreditLinesAndLoans': 2})]).iloc[0] == 0


def test_real_estate_loan_excluded():
    """부동산 담보대출이 있으면 제외"""
    assert _judge([_thin_filer_row(**{'NumberRealEstateLoansOrLines': 1})]).iloc[0] == 0


def test_revolving_utilization_not_used_for_judgement():
    """신용한도 이용률은 판정 기준이 아니다 (#15에서 제외)"""
    assert _judge([_thin_filer_row(
        **{'RevolvingUtilizationOfUnsecuredLines': 1.5})]).iloc[0] == 1


def test_debt_ratio_not_used_for_judgement():
    """부채비율은 판정 기준이 아니다 (#15에서 제외)"""
    assert _judge([_thin_filer_row(**{'DebtRatio': 2.0})]).iloc[0] == 1


def test_delinquency_history_excluded():
    """연체 기록이 있으면 제외"""
    assert _judge([_thin_filer_row(**{'NumberOfTimes90DaysLate': 1})]).iloc[0] == 0


def test_criteria1_missing_delinquency():
    """연체 컬럼 2개 이상 결측이면 기준 1로 씬파일러 (기준 2와 무관)"""
    assert _judge([_row(open_lines=10, real_estate=2,
                        d30=None, d60=None)]).iloc[0] == 1


def test_ratio_none_does_not_add_random_labels():
    """thin_filer_ratio=None이면 기준 해당자 수와 정확히 같아야 한다"""
    rows = [_thin_filer_row()] + [_row() for _ in range(99)]
    result = label_thin_filer(pd.DataFrame(rows), thin_filer_ratio=None)
    assert result['is_thin_filer'].sum() == 1


def test_ratio_forces_target_count():
    """thin_filer_ratio를 주면 목표 비율만큼 강제 조정된다 (평가용 아님)"""
    rows = [_thin_filer_row()] + [_row() for _ in range(99)]
    result = label_thin_filer(pd.DataFrame(rows), thin_filer_ratio=0.3, random_seed=42)
    assert result['is_thin_filer'].sum() == 30


def test_ratio_erases_delinquency_not_just_label():
    """손잡이(ratio)는 씬파일러 표시만 붙이지 않고 연체 칸을 실제로 지운다 (#61)"""
    rows = [_thin_filer_row()] + [_row() for _ in range(99)]
    result = label_thin_filer(pd.DataFrame(rows), thin_filer_ratio=0.3, random_seed=42)
    assert result['is_thin_filer'].sum() == 30
    erased = result[result['NumberOfTime30-59DaysPastDueNotWorse'].isna()]
    assert len(erased) == 29
    assert (erased['is_thin_filer'] == 1).all()


def test_ratio_result_is_reproducible_with_same_seed():
    """같은 시드면 같은 사람이 같은 방식으로 지워진다"""
    rows = [_thin_filer_row()] + [_row() for _ in range(99)]
    a = label_thin_filer(pd.DataFrame(rows), thin_filer_ratio=0.3, random_seed=7)
    b = label_thin_filer(pd.DataFrame(rows), thin_filer_ratio=0.3, random_seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_ratio_below_current_count_changes_nothing():
    """목표가 지금보다 낮으면 아무도 지우지 않는다"""
    rows = [_thin_filer_row() for _ in range(5)] + [_row() for _ in range(95)]
    result = label_thin_filer(pd.DataFrame(rows), thin_filer_ratio=0.01, random_seed=1)
    assert result['is_thin_filer'].sum() == 5
    assert result['NumberOfTime30-59DaysPastDueNotWorse'].notna().all()


def test_ratio_none_keeps_criteria_count():
    """None이면 기준 해당자 수 그대로 (지우기 없음)"""
    rows = [_thin_filer_row()] + [_row() for _ in range(99)]
    result = label_thin_filer(pd.DataFrame(rows), thin_filer_ratio=None)
    assert result['is_thin_filer'].sum() == 1
    assert result['NumberOfTime30-59DaysPastDueNotWorse'].notna().all()
