"""
Step 3: 성능 비교 · SHAP 설명 (남궁명진 담당)

학습된 모델과 Test 데이터로
  1) 성능 비교: AUC, KS, PSI, 혼동행렬, ROC·PR 곡선 → outputs/evaluation/metrics_<모델>.json
  2) SHAP: Summary Plot(상위 15개), Dependence Plot 2개, Waterfall 1개 → outputs/evaluation/shap/
  3) 거절 사유 Top 5 예시 → outputs/evaluation/explain_example.json
를 만든다.

실행 예:
  python main_step3.py --model models/xgboost_scale_pos_weight_v1.0.joblib
설정값(경로, 기준값)은 인자 또는 환경 변수로 바꾼다 (EVAL_OUTPUT_DIR, APPROVAL_THRESHOLD).
"""

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
DEFAULT_SPLITS = PROJECT_ROOT / 'data' / 'splits'
DEFAULT_RAW = PROJECT_ROOT / 'data' / 'intermediate' / '02_simulated.csv'
TARGET = 'SeriousDlqin2yrs'
ID_COL = 'Unnamed: 0'


def parse_args():
    p = argparse.ArgumentParser(description='성능 비교 · SHAP 설명')
    p.add_argument('--model', required=True, help='학습된 모델 파일 (.joblib / .pkl)')
    p.add_argument('--splits', default=os.getenv('SPLITS_DIR', str(DEFAULT_SPLITS)), help='train/test_processed.csv 폴더')
    p.add_argument('--raw', default=os.getenv('RAW_DATA', str(DEFAULT_RAW)), help='거절 사유 문장에 쓸 원래 값 CSV (선택)')
    p.add_argument('--out', default=os.getenv('EVAL_OUTPUT_DIR', str(PROJECT_ROOT / 'outputs' / 'evaluation')))
    p.add_argument('--shap-sample', type=int, default=int(os.getenv('SHAP_SAMPLE', '2000')), help='SHAP 계산에 쓸 Test 표본 수')
    return p.parse_args()


def model_features(model, columns):
    """모델이 학습한 칸 순서대로 고른다. 정보가 없으면 정답·보호 속성·표시 칸만 뺀다."""
    from src.explainer.feature_names import NON_FEATURE_COLS
    from src.explainer.shap_explainer import final_estimator
    names = getattr(final_estimator(model), 'feature_names_in_', None)
    if names is not None:
        missing = [c for c in names if c not in columns]
        if missing:
            raise ValueError(f"Test 파일에 모델이 학습한 칸이 없습니다: {missing}")
        return list(names)
    return [c for c in columns if c not in NON_FEATURE_COLS]


def main():
    import joblib
    import pandas as pd

    from src.explainer import performance as perf
    from src.explainer import shap_explainer as sx
    from src.explainer.feature_names import ALTERNATIVE_COLS, FINANCIAL_COLS
    from src.explainer.reasons import reference_values

    args = parse_args()
    out = Path(args.out)
    model_path = Path(args.model)
    model = joblib.load(model_path)
    logger.info(f"모델: {model_path.name}")

    train = pd.read_csv(Path(args.splits) / 'train_processed.csv')
    test = pd.read_csv(Path(args.splits) / 'test_processed.csv')
    features = model_features(model, list(test.columns))
    logger.info(f"입력 칸 {len(features)}개, Test {len(test):,}명")

    # 1) 성능 비교 (Test 는 여기서 한 번만 쓴다)
    p_test = model.predict_proba(test[features])[:, 1]
    p_train = model.predict_proba(train[features])[:, 1]
    result = perf.evaluate(test[TARGET], p_test, train_scores=p_train)
    result['model'] = model_path.name
    result['features'] = features
    path = perf.save_json(result, out / f"metrics_{model_path.stem}.json")
    logger.info(f"AUC {result['auc']:.4f} / KS {result['ks']:.4f} / PSI {result.get('psi', float('nan')):.4f} → {path}")
    if 'is_thin_filer' not in test.columns:
        logger.warning("Test 파일에 is_thin_filer 칸이 없어 씬파일러 AUC 향상은 계산하지 않았습니다.")

    # 2) SHAP
    sample = test.sample(n=min(args.shap_sample, len(test)), random_state=42)
    X = sample[features]
    explainer = sx.make_explainer(model, train[features])
    values = sx.shap_values_for_default(explainer, X)
    shap_dir = out / 'shap'
    sx.save_summary_plot(values, X, shap_dir / f"summary_{model_path.stem}.png")
    sx.save_dependence_plots(values, X, shap_dir)
    rejected = [i for i, p in enumerate(model.predict_proba(X)[:, 1]) if p >= result['confusion_matrix']['threshold']]
    idx = rejected[0] if rejected else 0
    sx.save_waterfall(explainer, values, X, idx, shap_dir / f"waterfall_{model_path.stem}.png")

    # 3) 거절 사유 예시: 원래 값(크기 맞추기 전)이 있으면 그것으로 문장을 만든다
    raw_row, refs = None, None
    raw_path = Path(args.raw)
    if raw_path.exists() and ID_COL in sample.columns:
        raw = pd.read_csv(raw_path)
        if ID_COL in raw.columns:
            raw = raw.set_index(ID_COL)
            cols = [c for c in FINANCIAL_COLS + ALTERNATIVE_COLS if c in raw.columns]
            train_ids = train[ID_COL] if ID_COL in train.columns else []
            refs = reference_values(raw.loc[raw.index.isin(train_ids), cols + ['age']].to_dict('records'), cols)
            raw_row = raw.loc[sample.iloc[idx][ID_COL]].to_dict()
    if raw_row is None:
        logger.warning("원래 값 파일과 연결할 번호 칸이 없어, 크기 맞춘 값으로 문장을 만듭니다 (예시용).")
        raw_row = X.iloc[idx].to_dict()
    explain = sx.explain_customer(explainer, X.iloc[[idx]], raw_row, references=refs)
    explain['disclaimer'] = perf.DISCLAIMER
    perf.save_json(explain, out / 'explain_example.json')
    logger.info("거절 사유 예시:")
    for r in explain['rejection_reasons']:
        logger.info(f"  {r['rank']}. {r['explanation']}")


if __name__ == '__main__':
    main()
