"""대시보드 본인 확인: 주민등록번호 형식 · 만 나이 · 성별 · 가리기 (src/dashboard/components/identity.py)"""
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dashboard"))
from components import identity  # noqa: E402

TODAY = date(2026, 11, 26)


@pytest.mark.parametrize("front, back, born, sex", [
    ("900101", "1234567", date(1990, 1, 1), "남성"),
    ("851231", "2234567", date(1985, 12, 31), "여성"),
    ("050315", "3234567", date(2005, 3, 15), "남성"),
    ("020704", "8234567", date(2002, 7, 4), "여성"),
])
def test_birth_date_and_gender(front, back, born, sex):
    assert identity.birth_date(front, back) == born
    assert identity.gender(back) == sex


def test_age_counts_birthday():
    assert identity.age("901126", "1234567", TODAY) == 36   # 오늘 생일
    assert identity.age("901127", "1234567", TODAY) == 35   # 내일 생일


@pytest.mark.parametrize("front, back", [("90010", "1234567"), ("900101", "123456"), ("9001a1", "1234567")])
def test_rrn_error_wrong_length(front, back):
    assert identity.rrn_error(front, back, TODAY) == "주민등록번호 13자리를 입력해 주세요"


def test_rrn_error_impossible_date_and_future():
    assert "생년월일" in identity.rrn_error("901332", "1234567", TODAY)
    assert "생년월일" in identity.rrn_error("301231", "3234567", TODAY)   # 2030년생


def test_rrn_error_minor():
    assert identity.rrn_error("100101", "3234567", TODAY) == "만 18세 이상 고객만 심사할 수 있어요"
    assert identity.rrn_error("900101", "1234567", TODAY) is None


def test_masked_keeps_only_first_back_digit():
    assert identity.masked("900101", "1234567") == "900101-1******"


def test_name_error():
    assert identity.name_error("  ") == "이름을 입력해 주세요"
    assert identity.name_error("홍길동2") == "이름을 다시 확인해 주세요"
    assert identity.name_error(" 홍  길동 ") is None
    assert identity.clean_name(" 홍  길동 ") == "홍 길동"


def test_digits_strips_separators():
    assert identity.digits("900101-1234567 ") == "9001011234567"
