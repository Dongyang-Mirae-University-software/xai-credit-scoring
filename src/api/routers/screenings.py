"""01 고객 심사: 심사 이력 저장·조회 (/screenings, POST /screenings/lookup)

SQLite 파일(SCREENINGS_DB, 기본 data/screenings.db)에 쌓아서 서버를 다시 띄워도 남는다."""
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from src.api.model import PROJECT_ROOT
from src.api.routers.customers import NOT_FOUND, find_by_identity
from src.api.schemas import CustomerLookup, ScreeningRecord

router = APIRouter(tags=['01 고객 심사'])


class ScreeningRequest(BaseModel):
    model_config = ConfigDict(extra='allow')
    input: dict
    result: dict


def _db():
    con = sqlite3.connect(os.getenv('SCREENINGS_DB', PROJECT_ROOT / 'data' / 'screenings.db'))
    con.row_factory = sqlite3.Row
    con.execute('CREATE TABLE IF NOT EXISTS screenings (id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT, '
                'approval_status TEXT, risk_grade TEXT, credit_score INTEGER, screened_at TEXT, input TEXT, result TEXT)')
    return con


def _sid(row_id):
    return f'S{row_id:04d}'


@router.post('/screenings', status_code=201, summary='심사 결과 저장',
             description='`/predict` 요청(input)과 응답(result)을 그대로 받아 저장한다. 고객 번호는 input.customer_id.')
def save_screening(body: ScreeningRequest):
    customer_id = str(body.input.get('customer_id') or body.result.get('customer_id') or '')
    screened_at = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    with closing(_db()) as con, con:
        cur = con.execute(
            'INSERT INTO screenings (customer_id, approval_status, risk_grade, credit_score, screened_at, input, result) '
            'VALUES (?,?,?,?,?,?,?)',
            (customer_id, body.result.get('approval_status'), body.result.get('risk_grade'), body.result.get('credit_score'),
             screened_at, json.dumps(body.input, ensure_ascii=False), json.dumps(body.result, ensure_ascii=False)))
    return {'screening_id': _sid(cur.lastrowid), 'screened_at': screened_at}


@router.get('/screenings', summary='최근 심사 목록',
            description='최신순. 같은 고객을 여러 번 심사했으면 가장 최근 1건만 나온다.')
def list_screenings(limit: int = Query(5, ge=1, le=100)):
    with closing(_db()) as con:
        rows = con.execute(
            'SELECT * FROM screenings WHERE id IN (SELECT MAX(id) FROM screenings GROUP BY customer_id) '
            'ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
    return [{'screening_id': _sid(r['id']), 'customer_id': r['customer_id'], 'approval_status': r['approval_status'],
             'risk_grade': r['risk_grade'], 'credit_score': r['credit_score'], 'screened_at': r['screened_at']}
            for r in rows]


NO_SCREENING = '이 고객의 심사 기록이 없습니다. 고객 심사를 먼저 해 주세요.'


@router.post('/screenings/lookup', response_model=ScreeningRecord, summary='고객의 가장 최근 심사 기록 (이름 + 주민등록번호)',
             description='본인 확인한 이름 · 주민등록번호로 그 고객의 가장 최근 심사를 돌려준다. '
                         '그때 심사에 넣은 값(input, `/predict` 요청)과 결과(result, `/predict` 응답)가 그대로 들어 있어, '
                         '`/explain` 에 input 을 넣으면 그때의 판단 근거를 다시 볼 수 있다 (XAI 분석 화면). '
                         '주민등록번호는 URL 에 남지 않도록 본문으로 받는다.',
             responses={404: {'description': '일치하는 고객 없음 / 심사 기록 없음 (detail 로 구분)'},
                        422: {'description': '주민등록번호 형식 오류'}})
def latest_screening(body: CustomerLookup):
    customer = find_by_identity(body.name, body.rrn)
    if customer is None:
        raise HTTPException(404, NOT_FOUND)
    with closing(_db()) as con:
        row = con.execute('SELECT * FROM screenings WHERE customer_id = ? ORDER BY id DESC LIMIT 1',
                          (customer['customer_id'],)).fetchone()
    if row is None:
        raise HTTPException(404, NO_SCREENING)
    return {'screening_id': _sid(row['id']), 'screened_at': row['screened_at'], 'customer_id': customer['customer_id'],
            'name': customer['name'], 'gender': customer.get('gender'),
            'input': json.loads(row['input']), 'result': json.loads(row['result'])}
