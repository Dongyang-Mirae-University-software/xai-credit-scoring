"""
Step 4: 공정성 검증 · Bias Mitigation (남궁명진 담당)

학습된 모델과 Test 데이터로
  1) 공정성 지표: 성별·연령대별 승인율, DI Ratio, Equalized Odds → outputs/fairness/fairness.json
  2) Mitigation 전/후 비교: Threshold Adjustment (+ 선택: Reweighting 재학습) → outputs/fairness/fairness_mitigation*.json
  3) 리포트용 전/후 비교 표 → outputs/fairness/comparison_*.md
를 만든다. JSON 형식은 대시보드 mock(src/dashboard/mock/fairness/)과 같다.

필요한 칸: Test 파일에 성별(gender 또는 Gender)과 연령대(age_group, 또는 원래 나이 age_raw).
분할 파일에 이 칸이 없으면 #33 머지 후 실행한다.

실행 예:
  python main_step4.py --model models/xgboost_scale_pos_weight_v1.0.joblib
  python main_step4.py --model models/xgboost_scale_pos_weight_v1.0.joblib --reweighting
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
DEFAULT_SPLITS = PROJECT_ROOT / 'data' / 'splits'
TARGET = 'SeriousDlqin2yrs'


def parse_args():
    p = argparse.ArgumentParser(description='공정성 검증 · Bias Mitigation')
    p.add_argument('--model', required=True, help='학습된 모델 파일 (.joblib / .pkl)')
    p.add_argument('--splits', default=os.getenv('SPLITS_DIR', str(DEFAULT_SPLITS)))
    p.add_argument('--out', default=os.getenv('FAIRNESS_OUTPUT_DIR', str(PROJECT_ROOT / 'outputs' / 'fairness')))
    p.add_argument('--reweighting', action='store_true', help='Reweighting 가중치로 모델을 다시 학습해 비교')
    return p.parse_args()


def group_columns(df):
    """성별·연령대 칸을 원본 기준 값(female/male, 20-34/35-54/55+)으로 만든다."""
    from src.fairness.metrics import normalize_age_group, normalize_gender
    gcol = next((c for c in ('gender', 'Gender') if c in df.columns), None)
    if gcol is None:
        raise SystemExit("Test 파일에 성별 칸(gender/Gender)이 없습니다. #33(분할 파일에 보호 속성 유지) 머지 후 실행하세요.")
    import pandas as pd
    # pandas 버전에 따라 글자 칸이 object 또는 문자열 형식으로 읽히므로, '숫자가 아닌지'로 확인한다
    if 'age_group' in df.columns and not pd.api.types.is_numeric_dtype(df['age_group']):
        age = normalize_age_group(values=df['age_group'])
    elif 'age_raw' in df.columns:
        age = normalize_age_group(ages=df['age_raw'])
    else:
        raise SystemExit("Test 파일에 원본 3구간 연령대(age_group) 또는 원래 나이(age_raw)가 없습니다. #33 머지 후 실행하세요.")
    return normalize_gender(df[gcol]), age


def main():
    import joblib
    import pandas as pd

    from src.explainer.feature_names import NON_FEATURE_COLS
    from src.explainer.performance import DISCLAIMER, save_json
    from src.explainer.shap_explainer import final_estimator
    from src.fairness import mitigation as mt
    from src.fairness.metrics import fairness_report

    args = parse_args()
    out = Path(args.out)
    model_path = Path(args.model)
    model = joblib.load(model_path)
    train = pd.read_csv(Path(args.splits) / 'train_processed.csv')
    test = pd.read_csv(Path(args.splits) / 'test_processed.csv')
    names = getattr(final_estimator(model), 'feature_names_in_', None)
    features = list(names) if names is not None else [c for c in test.columns if c not in NON_FEATURE_COLS + ['age_raw']]

    gender, age = group_columns(test)
    p = model.predict_proba(test[features])[:, 1]

    report = fairness_report(test[TARGET], p, gender=gender, age_group=age)
    report['model'] = model_path.name
    report['disclaimer'] = DISCLAIMER
    logger.info(f"DI Ratio {report['di_ratio']:.3f} / EO TPR 차이 {report['equalized_odds_tpr_diff']:.3f} "
                f"/ FPR 차이 {report['equalized_odds_fpr_diff']:.3f} → {save_json(report, out / 'fairness.json')}")

    ta = mt.mitigation_report(test[TARGET], p, gender=gender, age_group=age, method='threshold_adjustment')
    ta['disclaimer'] = DISCLAIMER
    save_json(ta, out / 'fairness_mitigation.json')
    (out / 'comparison_threshold_adjustment.md').write_text(mt.comparison_table(ta) + f"\n\n> {DISCLAIMER}\n", encoding='utf-8')
    logger.info("Threshold Adjustment 전/후:\n" + mt.comparison_table(ta))

    if args.reweighting:
        tr_gender, tr_age = group_columns(train)
        w = mt.reweighing_weights(train[TARGET], mt.combine_groups(tr_gender, tr_age))
        model_rw = mt.fit_with_weights(model, train[features], train[TARGET], w)
        p_rw = model_rw.predict_proba(test[features])[:, 1]
        rw = mt.mitigation_report(test[TARGET], p, gender=gender, age_group=age, method='reweighting', p_after=p_rw)
        rw['disclaimer'] = DISCLAIMER
        save_json(rw, out / 'fairness_mitigation_reweighting.json')
        (out / 'comparison_reweighting.md').write_text(mt.comparison_table(rw) + f"\n\n> {DISCLAIMER}\n", encoding='utf-8')
        logger.info("Reweighting 전/후:\n" + mt.comparison_table(rw))


if __name__ == '__main__':
    main()
