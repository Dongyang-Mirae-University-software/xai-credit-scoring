"""대시보드 공정성 페이지의 기준 판정 · 숫자 표시 (PR #130 리뷰)"""
import pytest

from src.dashboard.components.fairness_rules import di_ok, eo_ok, fmt_value
from src.fairness import metrics as fm

CRITERIA = {'di_min': fm.DI_MIN, 'di_max': fm.DI_MAX, 'eo_max': fm.EO_MAX}


@pytest.mark.parametrize('di', [0.79, 0.8, 0.996, 1.25, 1.2535])
def test_di_ok_matches_api_rule(di):
    # API 의 di_passed 와 같은 판정
    assert di_ok(di, CRITERIA) == (fm.DI_MIN <= di <= fm.DI_MAX)


def test_di_ok_without_upper_bound():
    assert di_ok(2.0, {'di_min': 0.8}) is True
    assert di_ok(None, CRITERIA) is None


def test_eo_ok():
    assert eo_ok(0.1, CRITERIA) is True
    assert eo_ok(0.148, CRITERIA) is False
    assert eo_ok(None, CRITERIA) is None


def test_fmt_value_shows_third_decimal_only_when_failing_near_bound():
    assert fmt_value(1.2535, False, CRITERIA) == '1.254'   # 1.25 로 보이면 통과처럼 읽힌다
    assert fmt_value(0.7951, False, CRITERIA) == '0.795'
    assert fmt_value(1.2535, True, CRITERIA) == '1.25'
    assert fmt_value(0.988, True, CRITERIA) == '0.99'
    assert fmt_value(0.148, False, CRITERIA) == '0.15'     # 경계값(0.10)과 멀면 둘째 자리
    assert fmt_value(None, None, CRITERIA) == '-'
