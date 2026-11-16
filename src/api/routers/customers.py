"""01 고객 심사: 고객 정보 조회 (/customers/{customer_id})

data/customers.json (scripts/add_dashboard_fields.py 가 만든 1000명). 값은 /predict 입력과 같은 단위다."""
import json
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from src.api.model import INCOME_KRW_PER_UNIT, PROJECT_ROOT

router = APIRouter(tags=['01 고객 심사'])


@lru_cache(maxsize=1)
def _customers():
    path = PROJECT_ROOT / 'data' / 'customers.json'
    if not path.exists():
        return {}
    # 파일은 모델 단위(원본 GMSC), 응답은 원
    return {c['customer_id']: {**c, 'MonthlyIncome': int(round(c['MonthlyIncome'] * INCOME_KRW_PER_UNIT))}
            for c in json.loads(path.read_text(encoding='utf-8'))}


@router.get('/customers/{customer_id}', summary='고객 정보 조회',
            description='`/predict` 입력 15개(+ 표시용 gender). 월소득은 원, 부채비율·납부율은 %, 신용한도 소진율은 0~1.',
            responses={404: {'description': '없는 고객'}})
def get_customer(customer_id: str):
    customer = _customers().get(customer_id.upper())
    if customer is None:
        raise HTTPException(404, f'고객을 찾을 수 없습니다: {customer_id}')
    return customer
