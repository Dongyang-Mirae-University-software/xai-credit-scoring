"""
Step 5: ANOVA 통계 검증 (남궁명진 담당)

MLflow 에 기록된 폴드별 AUC 로
  검증 1: 피처 그룹별 (전통 / 대안 / 통합) One-way ANOVA
  검증 2: 알고리즘별 (LR / XGB / LGBM, 통합 데이터) One-way ANOVA
를 계산하고, p < 0.05 이면 Tukey HSD 를 더해 outputs/statistics/anova.json 과 해석 문장(anova_interpretation.md)을 만든다.
JSON 형식은 대시보드 mock(src/dashboard/mock/statistical_test/anova.json)과 같다.

실행 예:
  python main_step5.py                         # MLflow 서버에서 읽기 (.env 의 MLFLOW_TRACKING_URI 등)
  python main_step5.py --runs-json runs.json   # MLflow REST runs/search 응답을 저장한 파일로 계산
"""

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description='ANOVA 통계 검증')
    p.add_argument('--runs-json', help='MLflow REST runs/search 응답 JSON 파일 (없으면 MLflow 서버에서 읽음)')
    p.add_argument('--experiment', default=os.getenv('MLFLOW_EXPERIMENT_NAME', 'xai-credit-scoring'))
    p.add_argument('--test2-data-type', default=os.getenv('ANOVA_TEST2_DATA_TYPE', 'integrated'))
    p.add_argument('--out', default=os.getenv('ANOVA_OUTPUT_DIR', str(PROJECT_ROOT / 'outputs' / 'statistics')))
    return p.parse_args()


def main():
    from src.statistics import anova

    args = parse_args()
    if args.runs_json:
        records = anova.records_from_rest(json.loads(Path(args.runs_json).read_text(encoding='utf-8')))
    else:
        records = anova.records_from_mlflow(args.experiment)
    logger.info(f"MLflow 기록 {len(records)}개 (조합별 최신 {len(anova.latest_per_combo(records))}개 사용)")

    report = anova.anova_report(records, test2_data_type=args.test2_data_type)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'anova.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# ANOVA 결과 해석', '']
    for key in ('test_1', 'test_2'):
        lines += [f"## {report[key]['test_name']}", ''] + [f"- {s}" for s in report['interpretation'][key]] + ['']
    lines.append(f"> {anova.DISCLAIMER}")
    (out / 'anova_interpretation.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for key in ('test_1', 'test_2'):
        for s in report['interpretation'][key]:
            logger.info(s)
    logger.info(f"저장: {out / 'anova.json'}, {out / 'anova_interpretation.md'}")


if __name__ == '__main__':
    main()
