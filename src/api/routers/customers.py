"""01 고객 심사: 고객 정보 조회 (/customers/{customer_id}, POST /customers/lookup)

data/customers.json (scripts/add_dashboard_fields.py 가 만든 1000명, 이름 · 주민등록번호는 scripts/add_customer_identity.py).
값은 /predict 입력과 같은 단위다. 주민등록번호는 조회 열쇠로만 쓰고 어떤 응답에도 넣지 않는다."""
import json
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from src.api.model import INCOME_KRW_PER_UNIT, PROJECT_ROOT
from src.api.schemas import CustomerLookup, CustomerProfile

router = APIRouter(tags=['01 고객 심사'])
NOT_FOUND = '일치하는 고객 정보가 없습니다. 이름과 주민등록번호를 확인해 주세요.'


@lru_cache(maxsize=1)
def _load():
    """(고객 ID → 응답, 주민등록번호 → 고객 ID). 파일은 모델 단위(원본 GMSC), 응답 월소득은 원."""
    path = PROJECT_ROOT / 'data' / 'customers.json'
    if not path.exists():
        return {}, {}
    by_id, by_rrn = {}, {}
    for c in json.loads(path.read_text(encoding='utf-8')):
        rrn = c.pop('rrn', None)
        by_id[c['customer_id']] = {**c, 'MonthlyIncome': int(round(c['MonthlyIncome'] * INCOME_KRW_PER_UNIT))}
        if rrn:
            by_rrn[rrn] = c['customer_id']
    return by_id, by_rrn


def _customers():
    return _load()[0]


@router.get('/customers/{customer_id}', summary='고객 정보 조회 (고객 번호)',
            description='`/predict` 입력 15개(+ 표시용 name · gender). 월소득은 원, 부채비율·납부율은 %, 신용한도 소진율은 0~1.',
            responses={404: {'description': '없는 고객'}})
def get_customer(customer_id: str):
    customer = _customers().get(customer_id.upper())
    if customer is None:
        raise HTTPException(404, f'고객을 찾을 수 없습니다: {customer_id}')
    return customer


@router.post('/customers/lookup', response_model=CustomerProfile, response_model_by_alias=True,
             summary='고객 정보 조회 (이름 + 주민등록번호)',
             description='창구에서 본인 확인한 이름 · 주민등록번호로 은행이 가진 신용정보 · 대안정보를 불러온다. '
                         '응답은 `/predict` 입력과 같은 단위라 그대로 심사에 넣을 수 있다. '
                         '주민등록번호와 이름이 모두 맞아야 하고, 어느 쪽이 틀렸는지는 알려 주지 않는다. '
                         '데이터의 이름 · 주민등록번호는 시연용 가짜 값이다.',
             responses={404: {'description': '일치하는 고객 없음'}, 422: {'description': '주민등록번호 형식 오류'}})
def lookup_customer(body: CustomerLookup):
    by_id, by_rrn = _load()
    customer = by_id.get(by_rrn.get(body.rrn))
    if customer is None or ' '.join(customer.get('name', '').split()) != body.name:
        raise HTTPException(404, NOT_FOUND)
    return customer
