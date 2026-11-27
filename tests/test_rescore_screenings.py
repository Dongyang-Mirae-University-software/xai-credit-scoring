"""scripts/rescore_screenings.py: 저장된 점수를 PDO 점수식으로 다시 계산 (이슈 #157)"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import rescore_screenings as rs  # noqa: E402
from src.api import model  # noqa: E402


def test_plan_rescores_only_old_rows():
    con = sqlite3.connect(':memory:')
    con.row_factory = sqlite3.Row
    con.execute('CREATE TABLE screenings (id INTEGER PRIMARY KEY, credit_score INTEGER, risk_grade TEXT, result TEXT)')
    con.executemany('INSERT INTO screenings VALUES (?,?,?,?)', [
        (1, 937, 'A', json.dumps({'default_probability': 0.063})),
        (2, 565, 'A', json.dumps({'default_probability': 0.063, 'score_method': model.SCORE_METHOD})),  # 이미 새 식
        (3, 700, 'B', json.dumps({})),  # 확률 없음 → 건너뜀
    ])
    rows = rs.plan(con)
    assert [(r[0], r[3], r[4]) for r in rows] == [(1, model.credit_score(0.063), 'A')]
    assert rows[0][5]['score_method'] == model.SCORE_METHOD
