"""04 모델 성능 (보너스 5): /ab-test"""
from fastapi import APIRouter

from src.api import reports
from src.api.schemas import AbTestResponse

router = APIRouter(tags=['04 모델 성능'])


@router.get('/ab-test', response_model=AbTestResponse, summary='A/B 테스트 (기존 CB 모델 vs 대안신용평가 모델)',
            description='Test 에서 정책 3개(같은 기준값 / 같은 승인율 / 같은 대손율) 비교, 무작위 A/B 1회 + 1,000회 반복, 필요 표본 수. '
                        '출처: outputs/evaluation/ab_test/ab_test.json (main_bonus_ab_test).')
def ab_test():
    return reports.load('ab_test')
