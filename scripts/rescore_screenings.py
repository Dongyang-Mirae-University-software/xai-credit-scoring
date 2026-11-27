"""저장된 심사 결과(SCREENINGS_DB)의 점수·등급을 지금 점수식(PDO, 이슈 #157)으로 다시 계산한다.

result JSON 의 default_probability 로 credit_score·risk_grade 를 다시 내고, result 에 approval_score·score_method 를 채운다.
승인 여부(approval_status)는 부도확률로 정한 것이라 바꾸지 않는다. score_method 가 이미 지금 값인 행은 건너뛴다.

    python scripts/rescore_screenings.py            # 기본: 바뀔 내용만 출력 (DB 그대로)
    python scripts/rescore_screenings.py --apply    # DB 파일을 같은 폴더에 백업한 뒤 반영
"""
import argparse
import json
import os
import shutil
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.api import model  # noqa: E402


def plan(con):
    """[(id, 이전 점수, 이전 등급, 새 점수, 새 등급, 새 result)] — 다시 계산할 행만."""
    rows = []
    for r in con.execute('SELECT id, credit_score, risk_grade, result FROM screenings ORDER BY id'):
        result = json.loads(r['result'] or '{}')
        prob = result.get('default_probability')
        if result.get('score_method') == model.SCORE_METHOD or prob is None:
            continue
        score = model.credit_score(float(prob))
        grade = model.risk_grade(score)
        new = {**result, 'credit_score': score, 'risk_grade': grade, 'approval_score': model.approval_score(),
               'score_method': model.SCORE_METHOD}
        rows.append((r['id'], r['credit_score'], r['risk_grade'], score, grade, new))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--apply', action='store_true', help='DB 를 백업한 뒤 실제로 반영 (없으면 출력만)')
    parser.add_argument('--db', default=os.getenv('SCREENINGS_DB', model.PROJECT_ROOT / 'data' / 'screenings.db'))
    args = parser.parse_args()
    db = Path(args.db)
    if not db.exists():
        sys.exit(f'DB 파일이 없습니다: {db}')

    with closing(sqlite3.connect(db)) as con:
        con.row_factory = sqlite3.Row
        total = con.execute('SELECT COUNT(*) FROM screenings').fetchone()[0]
        rows = plan(con)
        print(f'{db}: 전체 {total}건, 다시 계산할 행 {len(rows)}건 ({model.SCORE_METHOD})')
        for rid, old_s, old_g, s, g, _ in rows[:20]:
            print(f'  S{rid:04d}: {old_s}점 {old_g} → {s}점 {g}')
        if len(rows) > 20:
            print(f'  … 외 {len(rows) - 20}건')
        print('  등급 변화:', dict(Counter(f'{o}→{n}' for _, _, o, _, n, _ in rows)))
        if not args.apply:
            print('출력만 했습니다. 반영하려면 --apply')
            return
        backup = db.with_name(f'{db.stem}.backup-{datetime.now():%Y%m%d-%H%M%S}{db.suffix}')
        shutil.copy2(db, backup)
        with con:
            con.executemany('UPDATE screenings SET credit_score=?, risk_grade=?, result=? WHERE id=?',
                            [(s, g, json.dumps(new, ensure_ascii=False), rid) for rid, _, _, s, g, new in rows])
        print(f'반영 {len(rows)}건, 백업: {backup}')


if __name__ == '__main__':
    main()
