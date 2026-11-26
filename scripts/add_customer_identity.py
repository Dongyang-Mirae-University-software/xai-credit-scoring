"""
data/customers.json 고객 1000명에게 가짜 이름(name)과 주민등록번호(rrn)를 붙인다 (POST /customers/lookup 재료).

- 주민등록번호 앞자리(생년월일)는 기준일(REFERENCE_DATE)에 만 나이가 그 고객의 age 가 되도록 고른다
- 뒷자리 첫 숫자는 성별 · 출생 세기와 맞춘다 (1900년대 남 1 · 여 2, 2000년대 남 3 · 여 4)
- 마지막 검증 숫자는 일부러 틀리게 만든다 → 실제로는 존재할 수 없는 번호라 실존 인물과 겹치지 않는다
- 시드를 고정해 몇 번을 돌려도 같은 값이 나온다. 이미 붙어 있으면 다시 만든다 (다른 칸은 그대로)

실행: python scripts/add_customer_identity.py
"""
import json
import random
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'data' / 'customers.json'
SEED = 42
REFERENCE_DATE = date(2026, 11, 1)   # 이 날 기준으로 만 나이 = age

SURNAMES = ['김', '이', '박', '최', '정', '강', '조', '윤', '장', '임', '한', '오', '서', '신', '권', '황', '안', '송', '류', '홍']
GIVEN_A = ['민', '서', '지', '현', '준', '도', '하', '수', '예', '시', '주', '유', '은', '태', '승', '재', '동', '성', '영', '진']
GIVEN_B = ['준', '연', '우', '윤', '호', '아', '원', '빈', '은', '희', '훈', '진', '혁', '나', '린', '석', '민', '경', '현', '율']
CHECK_WEIGHTS = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]


def check_digit(first12):
    """주민등록번호 검증 숫자 (앞 12자리 가중합)."""
    return (11 - sum(int(d) * w for d, w in zip(first12, CHECK_WEIGHTS)) % 11) % 10


def birth_date(rng, age):
    """기준일에 만 나이가 age 인 생일."""
    month, day = rng.randint(1, 12), rng.randint(1, 28)
    year = REFERENCE_DATE.year - age - ((month, day) > (REFERENCE_DATE.month, REFERENCE_DATE.day))
    return date(year, month, day)


def fake_rrn(rng, age, gender):
    born = birth_date(rng, age)
    male = gender == '남성'
    first = (1 if male else 2) if born.year < 2000 else (3 if male else 4)
    first12 = f'{born:%y%m%d}{first}{rng.randint(0, 99999):05d}'
    return first12 + str((check_digit(first12) + 1) % 10)   # 일부러 틀린 검증 숫자


def main():
    rng = random.Random(SEED)
    customers = json.loads(PATH.read_text(encoding='utf-8'))
    used = set()
    for c in sorted(customers, key=lambda c: c['customer_id']):   # 파일 순서가 바뀌어도 같은 고객은 같은 값
        c['name'] = rng.choice(SURNAMES) + rng.choice(GIVEN_A) + rng.choice(GIVEN_B)
        rrn = fake_rrn(rng, c['age'], c['gender'])
        while rrn in used:
            rrn = fake_rrn(rng, c['age'], c['gender'])
        used.add(rrn)
        c['rrn'] = rrn
    PATH.write_text(json.dumps(customers, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    sample = next(c for c in customers if c['customer_id'] == 'C100001')
    print(f"{len(customers)}명에 이름 · 주민등록번호를 붙였어요. 예: {sample['customer_id']} {sample['name']} {sample['rrn']}")


if __name__ == '__main__':
    main()
