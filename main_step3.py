"""
Step 3: 성능 비교 · SHAP 설명 (남궁명진 담당)

학습된 모델과 Test 데이터로
  1) 성능 비교: AUC, KS, PSI(점수·특성별), 혼동행렬, ROC·PR 곡선 → outputs/evaluation/metrics_<모델>.json
     --baseline-model(금융만 쓴 모델)을 주면 씬파일러 AUC 향상과 신뢰구간도 계산한다 (#27)
  2) SHAP: Summary Plot(상위 15개), Dependence Plot 2개, Waterfall 1개 → outputs/evaluation/shap/
  3) 거절 사유 Top 5 예시 → outputs/evaluation/explain_example.json
     (+ API 재료: shap_summary.json, reference_values.json)
를 만든다.

실행 예:
  python main_step3.py --model models/lightgbm_is_unbalance_integrated_v1.0.joblib
  python main_step3.py --model models/lightgbm_is_unbalance_integrated_v1.0.joblib \
      --baseline-model models/lightgbm_is_unbalance_financial_only_v1.0.joblib   (main_step2 실행 후)
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
SPLIT_ID = 'customer_id'   # 분할 파일의 고객 번호 (#58)
RAW_ID = 'Unnamed: 0'      # 02_simulated.csv의 같은 번호


def parse_args():
    p = argparse.ArgumentParser(description='성능 비교 · SHAP 설명')
    p.add_argument('--model', required=True, help='학습된 모델 파일 (.joblib / .pkl)')
    p.add_argument('--splits', default=os.getenv('SPLITS_DIR', str(DEFAULT_SPLITS)), help='train/test_processed.csv 폴더')
    p.add_argument('--raw', default=os.getenv('RAW_DATA', str(DEFAULT_RAW)), help='거절 사유 문장에 쓸 원래 값 CSV (선택)')
    p.add_argument('--out', default=os.getenv('EVAL_OUTPUT_DIR', str(PROJECT_ROOT / 'outputs' / 'evaluation')))
    p.add_argument('--shap-sample', type=int, default=int(os.getenv('SHAP_SAMPLE', '2000')), help='SHAP 계산에 쓸 Test 표본 수')
    p.add_argument('--baseline-model', default=os.getenv('BASELINE_MODEL'),
                   help='씬파일러 AUC 향상 비교용, 대안 데이터를 쓰지 않은 모델 파일 (선택)')
    p.add_argument('--allow-scaled', action='store_true',
                   help='원래 값과 연결되지 않을 때도 표준화된 값으로 거절 사유 예시를 만든다 (예시 확인용, 문장의 숫자가 실제 값이 아님)')
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
    result['feature_psi'] = perf.feature_psi(train[features], test[features], features)
    if not args.baseline_model:
        logger.info("--baseline-model 이 없어 씬파일러 AUC 향상은 계산하지 않았습니다 (main_step2 결과에는 기록됨).")
    elif 'is_thin_filer' not in test.columns:
        logger.warning("Test 파일에 is_thin_filer 칸이 없어 씬파일러 AUC 향상은 계산하지 않았습니다.")
    else:
        baseline = joblib.load(args.baseline_model)
        p_base = baseline.predict_proba(test[model_features(baseline, list(test.columns))])[:, 1]
        uplift = perf.thin_filer_uplift_ci(test[TARGET], p_base, p_test, test['is_thin_filer'])
        uplift['baseline_model'] = Path(args.baseline_model).name
        result['thin_filer_uplift'] = uplift
        result['pass']['thin_filer_uplift'] = uplift['pass']
        logger.info(f"씬파일러 AUC 향상 {uplift['uplift']:+.4f} [{uplift['ci_low']:+.4f} ~ {uplift['ci_high']:+.4f}] "
                    f"(씬파일러 {uplift['n_thin_filer']}명, 부도 {uplift['n_thin_filer_default']}명)")
    path = perf.save_json(result, out / f"metrics_{model_path.stem}.json")
    fp = result['feature_psi']
    logger.info(f"AUC {result['auc']:.4f} / KS {result['ks']:.4f} / PSI {result.get('psi', float('nan')):.4f} "
                f"(특성 최대 {fp['max']:.4f} {fp['max_feature']}) → {path}")

    # 2) SHAP
    sample = test.sample(n=min(args.shap_sample, len(test)), random_state=42)
    X = sample[features]
    explainer = sx.make_explainer(model, train[features])
    values = sx.shap_values_for_default(explainer, X)
    shap_dir = out / 'shap'
    sx.save_summary_plot(values, X, shap_dir / f"summary_{model_path.stem}.png")
    # API GET /explain/summary 재료: 변수별 mean |SHAP| (전통/대안 구분)
    mean_abs = abs(values).mean(axis=0)
    perf.save_json({
        'model': model_path.name, 'n_samples': len(X), 'shap_space': 'log-odds',
        'features': sorted(({'feature': f, 'mean_abs_shap': float(v),
                             'group': 'alternative' if f in ALTERNATIVE_COLS else 'financial'}
                            for f, v in zip(features, mean_abs)), key=lambda d: -d['mean_abs_shap']),
        'disclaimer': perf.DISCLAIMER,
    }, out / 'shap_summary.json')
    sx.save_dependence_plots(values, X, shap_dir)
    rejected = [i for i, p in enumerate(model.predict_proba(X)[:, 1]) if p >= result['confusion_matrix']['threshold']]
    idx = rejected[0] if rejected else 0
    sx.save_waterfall(explainer, values, X, idx, shap_dir / f"waterfall_{model_path.stem}.png")

    # 3) 거절 사유 예시: 원래 값(크기 맞추기 전)이 있으면 그것으로 문장을 만든다
    raw_row, refs = None, None
    raw_path = Path(args.raw)
    if raw_path.exists() and SPLIT_ID in sample.columns:
        raw = pd.read_csv(raw_path)
        raw_id = RAW_ID if RAW_ID in raw.columns else (SPLIT_ID if SPLIT_ID in raw.columns else None)
        train_ids = train[SPLIT_ID] if SPLIT_ID in train.columns else []
        # 전처리(#54, #81)와 같게: 월소득이 비어 있는 행의 DebtRatio는 금액이므로
        # 학습(Train) 행의 월소득 중앙값으로 나눠 비율로 맞춘다 (학습 행을 못 찾으면 전체 중앙값)
        if 'MonthlyIncome' in raw.columns and 'DebtRatio' in raw.columns:
            income_missing = raw['MonthlyIncome'].isna()
            is_train = raw[raw_id].isin(train_ids) if raw_id is not None else pd.Series(False, index=raw.index)
            income_median = raw.loc[is_train, 'MonthlyIncome'].median() if is_train.any() else raw['MonthlyIncome'].median()
            raw.loc[income_missing, 'DebtRatio'] = raw.loc[income_missing, 'DebtRatio'] / income_median
            logger.info(f"DebtRatio 보정: 월소득 결측 {income_missing.sum():,}행 ÷ 월소득 중앙값 {income_median:,.0f}")
        if raw_id is not None:
            raw = raw.set_index(raw_id)
            cols = [c for c in FINANCIAL_COLS + ALTERNATIVE_COLS if c in raw.columns]
            # age는 FINANCIAL_COLS에도 있어서, 연령대 구분용으로 한 번만 넣는다
            ref_cols = cols if 'age' in cols else cols + ['age']
            refs = reference_values(raw.loc[raw.index.isin(train_ids), ref_cols].to_dict('records'), cols)
            perf.save_json(refs, out / 'reference_values.json')  # API POST /explain 의 '동일 연령대 평균'
            customer = sample.iloc[idx][SPLIT_ID]
            if customer in raw.index:
                raw_row = raw.loc[customer].to_dict()
    scaled = False
    if raw_row is None:
        msg = (f"거절 사유에 쓸 원래 값을 찾지 못했습니다 (분할 파일의 {SPLIT_ID} 칸 또는 {raw_path} 확인). "
               "지금 코드로 main_step1.py를 다시 실행해 분할 파일을 만들거나(#32), "
               "예시만 볼 때는 --allow-scaled 를 붙이세요.")
        if not args.allow_scaled:
            raise SystemExit(msg)
        logger.warning(msg + " --allow-scaled: 표준화된 값으로 예시를 만듭니다 (문장의 숫자는 실제 값이 아님).")
        raw_row = X.iloc[idx].to_dict()
        scaled = True
    explain = sx.explain_customer(explainer, X.iloc[[idx]], raw_row, references=refs)
    explain['values_are_scaled'] = scaled
    explain['disclaimer'] = perf.DISCLAIMER
    perf.save_json(explain, out / 'explain_example.json')
    logger.info("거절 사유 예시:")
    for r in explain['rejection_reasons']:
        logger.info(f"  {r['rank']}. {r['explanation']}")


if __name__ == '__main__':
    main()
