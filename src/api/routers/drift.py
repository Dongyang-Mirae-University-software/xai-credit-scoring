"""06 모니터링: /drift"""
from fastapi import APIRouter

from src.api import reports
from src.api.schemas import DriftResponse

router = APIRouter(tags=['06 모니터링'])


@router.get('/drift', response_model=DriftResponse, summary='데이터 드리프트 (PSI · CSI)',
            description='Train 기준 시나리오별 점수 PSI, 칸별 CSI, AUC·승인율. 구간별 상세(tables)는 include_tables=true 일 때만 포함 '
                        '(내장 예시에는 tables 가 없다). 출처: outputs/monitoring/drift_report.json (main_bonus_drift).')
def drift(include_tables: bool = False):
    report = reports.load('drift')
    if not include_tables:
        for s in report['scenarios'].values():
            s.pop('tables', None)
    return report
