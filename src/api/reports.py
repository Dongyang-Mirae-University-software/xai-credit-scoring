"""
파이프라인 산출물(JSON) 읽기.

main_step3~5 가 outputs/ 아래에 저장한 파일이 있으면 그것을, 없으면 src/api/examples/ 의 예시를 돌려준다.
그래서 실제 데이터가 생기면 파일만 그 자리에 두면 되고, API 코드는 바꾸지 않는다.
OUTPUTS_DIR 환경 변수로 폴더를 바꿀 수 있다.
"""
import json
import os
from pathlib import Path

from src.api.model import MODEL_VERSION, PROJECT_ROOT

OUTPUTS_DIR = Path(os.getenv('OUTPUTS_DIR', PROJECT_ROOT / 'outputs'))
EXAMPLES_DIR = Path(__file__).parent / 'examples'

# 리포트 이름 → 파이프라인이 저장하는 위치 (outputs/ 기준)
PATHS = {
    'metrics': f'evaluation/metrics_{MODEL_VERSION}.json',       # main_step3
    'shap_summary': 'evaluation/shap_summary.json',              # main_step3
    'reference_values': 'evaluation/reference_values.json',      # main_step3 (거절 사유 '동일 연령대 평균')
    'fairness': 'fairness/fairness.json',                        # main_step4
    'fairness_mitigation': 'fairness/fairness_mitigation.json',  # main_step4
    'anova': 'statistics/anova.json',                            # main_step5
}


def path(name):
    return OUTPUTS_DIR / PATHS[name]


def source(name):
    """/health 표시용: 실제 파일 경로 또는 'example'."""
    p = path(name)
    if p.exists():
        return str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)
    return 'example' if (EXAMPLES_DIR / f'{name}.json').exists() else 'missing'


def load(name, fallback=True):
    """산출물 JSON. 없으면 예시(fallback=True) 또는 None."""
    p = path(name)
    if not p.exists():
        p = EXAMPLES_DIR / f'{name}.json'
        if not fallback or not p.exists():
            return None
    return json.loads(p.read_text(encoding='utf-8'))


def model_metadata():
    return json.loads((PROJECT_ROOT / 'models' / 'model_metadata.json').read_text(encoding='utf-8'))
