"""본인 확인: 이름 · 주민등록번호 형식 확인, 만 나이 · 성별 계산, 화면 표시용 가리기.

주민등록번호는 정보 조회 요청(POST /customers/lookup)에만 쓴다. 심사 요청(/predict) · 심사 이력에는 넣지 않고,
화면에는 뒷자리 첫 숫자만 남긴 값(900101-1******)만 보여 준다.
"""
import re
from datetime import date

MIN_AGE = 18  # API(/predict) 가 받는 최소 나이
NAME_MAX = 20
_DIGITS = re.compile(r"\d+")

# 뒷자리 첫 숫자 → (출생 세기, 성별). 1·2 / 5·6 은 1900년대, 3·4 / 7·8 은 2000년대, 9·0 은 1800년대 (5~8 외국인)
_CENTURY = {
    "1": (1900, "남성"), "2": (1900, "여성"), "3": (2000, "남성"), "4": (2000, "여성"),
    "5": (1900, "남성"), "6": (1900, "여성"), "7": (2000, "남성"), "8": (2000, "여성"),
    "9": (1800, "남성"), "0": (1800, "여성"),
}


def clean_name(name):
    return " ".join((name or "").split())


def name_error(name):
    """이름 칸 안내 (문제없으면 None)."""
    name = clean_name(name)
    if not name:
        return "이름을 입력해 주세요"
    if len(name) > NAME_MAX or _DIGITS.search(name):
        return "이름을 다시 확인해 주세요"
    return None


def digits(text):
    """입력칸 값에서 숫자만 남긴다 (붙여 넣은 '-' 나 공백은 지운다)."""
    return "".join(_DIGITS.findall(text or ""))


def birth_date(front, back):
    """생년월일. 형식이 틀리거나 없는 날짜면 ValueError."""
    if len(front) != 6 or len(back) != 7 or not (front + back).isdigit():
        raise ValueError("주민등록번호 13자리를 입력해 주세요")
    century, _ = _CENTURY[back[0]]
    try:
        return date(century + int(front[:2]), int(front[2:4]), int(front[4:6]))
    except ValueError:
        raise ValueError("주민등록번호 앞자리(생년월일)를 다시 확인해 주세요") from None


def gender(back):
    return _CENTURY[back[0]][1]


def age(front, back, today=None):
    """만 나이 (생일이 지나야 한 살 더한다)."""
    born, today = birth_date(front, back), today or date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def rrn_error(front, back, today=None):
    """주민등록번호 칸 안내 (문제없으면 None)."""
    try:
        born = birth_date(front, back)
    except ValueError as e:
        return str(e)
    today = today or date.today()
    if born > today:
        return "주민등록번호 앞자리(생년월일)를 다시 확인해 주세요"
    if age(front, back, today) < MIN_AGE:
        return f"만 {MIN_AGE}세 이상 고객만 심사할 수 있어요"
    return None


def masked(front, back):
    """화면 표시용: 900101-1******"""
    return f"{front}-{back[:1]}{'*' * 6}"
