"""
Step 6: 검증 결과를 쉬운 영어로 요약하고 MLflow final_test run 에 모은다 (남궁명진 담당)

main_step3(성능·SHAP)·main_step4(공정성)·main_step5(ANOVA)가 만든 파일을 읽어
  1) outputs/mlflow_summary/summary_en.md   결과 요약 (쉬운 영어)
  2) outputs/mlflow_summary/metrics_guide.md 지표·파라미터 이름표 (쉬운 영어)
를 만들고, --upload 를 주면 가장 최근 final_test run 에 위 파일과 결과 파일·그림을 올리고
run 설명(Description)을 쉬운 영어로 쓴다. --upload 가 없으면 무엇을 올릴지만 보여 준다.

실행 예:
  python main_step6.py                 # 요약만 만들고 올릴 목록 확인
  python main_step6.py --upload        # .env 의 MLflow 서버에 올림 (서버 기록이 바뀜)
"""

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
OUTPUTS = PROJECT_ROOT / 'outputs'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='검증 결과 쉬운 영어 요약 + MLflow final_test 에 올리기')
    p.add_argument('--eval-dir', default=os.getenv('EVAL_OUTPUT_DIR', str(OUTPUTS / 'evaluation')))
    p.add_argument('--fairness-dir', default=os.getenv('FAIRNESS_OUTPUT_DIR', str(OUTPUTS / 'fairness')))
    p.add_argument('--stats-dir', default=os.getenv('ANOVA_OUTPUT_DIR', str(OUTPUTS / 'statistics')))
    p.add_argument('--out', default=os.getenv('MLFLOW_SUMMARY_DIR', str(OUTPUTS / 'mlflow_summary')))
    p.add_argument('--upload', action='store_true', help='MLflow 서버의 final_test run 에 올린다 (서버 기록이 바뀜)')
    p.add_argument('--run-id', help='올릴 run (없으면 가장 최근 final_test)')
    p.add_argument('--env-file', default=str(PROJECT_ROOT / '.env'))
    return p.parse_args(argv)


def _load(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def build(eval_dir, fairness_dir, stats_dir, out_dir):
    """요약 파일을 만들고 올릴 목록 [(로컬 파일, MLflow 안 폴더)] 을 돌려준다."""
    from src.tracking import notes

    eval_dir, fairness_dir, stats_dir, out_dir = map(Path, (eval_dir, fairness_dir, stats_dir, out_dir))
    metrics_files = sorted(eval_dir.glob('metrics_*.json'), key=lambda p: p.stat().st_mtime)
    perf = _load(metrics_files[-1]) if metrics_files else None
    explain = _load(eval_dir / 'explain_example.json')
    fairness = _load(fairness_dir / 'fairness.json')
    mitigation = _load(fairness_dir / 'fairness_mitigation.json')
    anova = _load(stats_dir / 'anova.json')
    if not any((perf, explain, fairness, anova)):
        raise SystemExit("요약할 결과 파일이 없습니다. main_step3·main_step4·main_step5 를 먼저 실행하세요.")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / 'summary_en.md'
    summary.write_text(notes.summary_md(perf, explain, fairness, mitigation, anova), encoding='utf-8')
    guide = out_dir / 'metrics_guide.md'
    guide.write_text(notes.metrics_guide_md(), encoding='utf-8')

    plan = [(summary, ''), (guide, '')]
    if metrics_files:
        plan.append((metrics_files[-1], 'evaluation'))
    for f in [eval_dir / 'explain_example.json'] + sorted((eval_dir / 'shap').glob('*.png')):
        if f.exists():
            plan.append((f, 'evaluation/shap' if f.suffix == '.png' else 'evaluation'))
    for pattern in ('fairness*.json', 'comparison_*.md'):
        plan += [(f, 'fairness') for f in sorted(fairness_dir.glob(pattern))]
    for name in ('anova.json', 'anova_interpretation.md'):
        if (stats_dir / name).exists():
            plan.append((stats_dir / name, 'statistics'))
    return plan, {'perf': perf, 'fairness': fairness, 'anova': anova}


def upload(plan, env_file, run_id=None, client=None):
    """final_test run 에 파일을 올리고 설명·태그를 쉬운 영어로 남긴다."""
    from src.tracking import notes
    from src.tracking.rest import MlflowRest, read_env

    client = client or MlflowRest.from_env(env_file)
    if run_id:
        run = client.get_run(run_id)
    else:
        exp_name = read_env(env_file).get('MLFLOW_EXPERIMENT_NAME', 'xai-credit-scoring')
        run = client.latest_run(client.experiment_id(exp_name), 'final_test')
    info, data = run['info'], run.get('data', {})
    rid = info['run_id']
    logger.info(f"올릴 run: {info.get('run_name')} ({rid})")
    for path, folder in plan:
        logger.info(f"  올림: {client.upload(info['artifact_uri'], path, folder)}")

    metrics = {m['key']: m['value'] for m in data.get('metrics', [])}
    tags = {t['key']: t['value'] for t in data.get('tags', [])}
    if 'test_auc' in metrics and 'test_ks' in metrics:
        uplift = None
        if all(k in metrics for k in ('thin_uplift', 'thin_ci_low', 'thin_ci_high')):
            uplift = {'uplift': metrics['thin_uplift'], 'ci_low': metrics['thin_ci_low'], 'ci_high': metrics['thin_ci_high']}
        note = notes.final_test_note(tags.get('selected_model', 'final model'), metrics['test_auc'], metrics['test_ks'],
                                     psi=metrics.get('test_psi'), uplift=uplift)
        client.set_tag(rid, 'mlflow.note.content', note + ' See summary_en.md and metrics_guide.md in Artifacts.')
    client.set_tag(rid, 'results_attached', 'SHAP plots, fairness and ANOVA results with a plain-English summary')
    return rid


def main(argv=None):
    args = parse_args(argv)
    plan, _ = build(args.eval_dir, args.fairness_dir, args.stats_dir, args.out)
    logger.info(f"쉬운 영어 요약: {Path(args.out) / 'summary_en.md'}")
    logger.info("올릴 파일:")
    for path, folder in plan:
        logger.info(f"  {folder or '(맨 위)'} ← {path}")
    if not args.upload:
        logger.info("--upload 가 없어 MLflow 서버에는 올리지 않았습니다.")
        return
    upload(plan, args.env_file, run_id=args.run_id)
    logger.info("✅ final_test run 에 올림")


if __name__ == '__main__':
    main()
