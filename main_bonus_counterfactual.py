"""
보너스 과제 2: Counterfactual Explanation (과제 노션 「5. 보너스 과제 (선택)」 2번)

Test 에서 거절된 고객에게 "어떤 조건이 바뀌면 승인되는지"를 찾아 본다 (src/explainer/counterfactual.py).
  - 표본: Test 거절 고객(부도 확률 ≥ 승인 기준) 중 무작위 300명 (seed 42)
  - 결과: outputs/evaluation/counterfactual/counterfactual_summary.json, counterfactual.png
같은 함수를 API POST /counterfactual 이 고객 한 명에게 쓴다.

사용법
  python main_bonus_counterfactual.py --model models/lightgbm_is_unbalance_integrated_v1.0.joblib
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
DEFAULT_SPLITS = PROJECT_ROOT / 'data' / 'splits'
DEFAULT_SCALER = PROJECT_ROOT / 'models' / 'preprocess_scaler_v1.0.joblib'


def parse_args():
    p = argparse.ArgumentParser(description='보너스 2: Counterfactual Explanation')
    p.add_argument('--model', required=True, help='학습된 모델 파일 (.joblib / .pkl)')
    p.add_argument('--scaler', default=os.getenv('SCALER_PATH', str(DEFAULT_SCALER)))
    p.add_argument('--splits', default=os.getenv('SPLITS_DIR', str(DEFAULT_SPLITS)))
    p.add_argument('--out', default=os.getenv('CF_OUTPUT_DIR', str(PROJECT_ROOT / 'outputs' / 'evaluation' / 'counterfactual')))
    p.add_argument('--sample', type=int, default=int(os.getenv('CF_SAMPLE', '300')), help='거절 고객 표본 수')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def check_bounds(train_raw):
    """ACTIONABLE 의 분위수가 지금 Train 과 맞는지 확인 (데이터가 바뀌면 경고)."""
    from src.explainer.counterfactual import ACTIONABLE
    for f, (_, low, high, _) in ACTIONABLE.items():
        q1, q99 = float(train_raw[f].quantile(0.01)), float(train_raw[f].quantile(0.99))
        if abs(q1 - low) > 1e-3 * max(1, abs(low)) or abs(q99 - high) > 1e-3 * max(1, abs(high)):
            logger.warning("%s 분위수가 바뀌었습니다: 코드 (%.4f, %.4f) / Train (%.4f, %.4f)", f, low, high, q1, q99)


def save_plot(single_rate, combined_freq, success, path):
    from src.explainer import shap_explainer as sx
    plt = sx._setup_font()
    feats = sorted(single_rate, key=lambda f: -(single_rate[f] + combined_freq[f]))
    names = [sx._label(f) for f in feats]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    ax1.barh(names, [single_rate[f] for f in feats], color='#5B6CD9')
    ax1.set_xlim(0, 1)
    ax1.set_title('이 칸 하나만 바꿔도 승인되는 고객 비율')
    ax2.barh(names, [combined_freq[f] for f in feats], color='#2A9D8F')
    ax2.set_xlim(0, 1)
    ax2.set_title(f'여러 칸 제안에 들어간 비율 (승인 가능 {success:.0%})')
    ax1.invert_yaxis()
    for ax in (ax1, ax2):
        ax.set_xlabel('거절 고객 중 비율')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    import joblib
    import numpy as np
    import pandas as pd

    from src.explainer import counterfactual as cf
    from src.explainer.feature_names import korean_name
    from src.explainer.performance import DISCLAIMER, save_json

    model, scaler = joblib.load(args.model), joblib.load(args.scaler)
    cols = list(scaler.feature_names_in_)
    model_cols = list(getattr(model, 'feature_names_in_', cols))
    threshold = float(os.getenv('APPROVAL_THRESHOLD', '0.5'))

    train = pd.read_csv(Path(args.splits) / 'train_processed.csv')
    test = pd.read_csv(Path(args.splits) / 'test_processed.csv')
    check_bounds(pd.DataFrame(scaler.inverse_transform(train[cols]), columns=cols))

    def predict_proba(raw):
        X = pd.DataFrame(scaler.transform(raw[cols].astype(float)), columns=cols)
        return model.predict_proba(X[model_cols])[:, 1]

    proba = model.predict_proba(test[model_cols])[:, 1]
    rejected = test[proba >= threshold]
    sample = rejected.sample(n=min(args.sample, len(rejected)), random_state=args.seed)
    raw = pd.DataFrame(scaler.inverse_transform(sample[cols]), columns=cols, index=sample.index)
    logger.info("Test 거절 %d명 중 %d명 분석", len(rejected), len(sample))

    records = []
    for idx, x in raw.iterrows():
        x = x.to_dict()
        single = cf.single_changes(x, predict_proba, threshold)
        comb = cf.combined_changes(x, predict_proba, threshold)
        records.append({
            'customer_id': int(sample.at[idx, 'customer_id']) if 'customer_id' in sample else int(idx),
            'is_thin_filer': bool(sample.at[idx, 'is_thin_filer']) if 'is_thin_filer' in sample else None,
            'probability': float(predict_proba(pd.DataFrame([x]))[0]),
            'x': x, 'single': single, 'combined': comb,
        })

    n = len(records)
    feats = list(cf.ACTIONABLE)
    ok = [r for r in records if r['combined']['success']]
    single_rate = {f: sum(1 for r in records if r['single'][f]) / n for f in feats}
    combined_freq = {f: sum(1 for r in ok if f in r['combined']['changes']) / max(len(ok), 1) for f in feats}
    any_single = sum(1 for r in records if any(r['single'].values())) / n

    def med_change(f):
        mult = cf.DISPLAY_UNITS[f][0]
        d = [(r['combined']['changes'][f] - r['x'][f]) * mult for r in ok if f in r['combined']['changes']]
        return float(np.median(d)) if d else None

    def group(rs):
        good = [r for r in rs if r['combined']['success']]
        return {'n': len(rs), 'combined_success_rate': len(good) / len(rs) if rs else None,
                'mean_changed_features': float(np.mean([len(r['combined']['changes']) for r in good])) if good else None}

    examples = []
    for r in ok[:3]:
        examples.append({
            'customer_id': r['customer_id'], 'probability': round(r['probability'], 4),
            'new_probability': round(r['combined']['new_probability'], 4),
            'changes': [cf.describe(f, r['x'][f], v, cf.DISPLAY_UNITS) for f, v in r['combined']['changes'].items()],
        })

    summary = {
        'task': '보너스 2: Counterfactual Explanation (거절 고객이 승인되려면 무엇이 바뀌어야 하나)',
        'model': Path(args.model).name, 'approval_threshold': threshold, 'seed': args.seed,
        'data': f'Test 거절 고객 {len(rejected)}명 중 무작위 {n}명',
        'actionable_features': {f: {'name_kr': korean_name(f), 'direction': '증가' if d > 0 else '감소',
                                    'train_q01': lo, 'train_q99': hi, 'max_relative_change': mr}
                                for f, (d, lo, hi, mr) in cf.ACTIONABLE.items()},
        'immutable_features': cf.IMMUTABLE,
        'combined': {**group(records),
                     'mean_new_probability': float(np.mean([r['combined']['new_probability'] for r in ok])) if ok else None,
                     'mean_normalized_distance': float(np.mean([cf.normalized_distance(r['x'], r['combined']['changes']) for r in ok])) if ok else None,
                     'feature_frequency': combined_freq,
                     'median_change_display_units': {f: med_change(f) for f in feats}},
        'single': {'any_single_feature_rate': any_single, 'rate_by_feature': single_rate},
        'by_segment': {'thin_filer': group([r for r in records if r['is_thin_filer']]),
                       'thick_filer': group([r for r in records if r['is_thin_filer'] is False])},
        'failed_mean_probability': float(np.mean([r['probability'] for r in records if not r['combined']['success']]))
        if len(ok) < n else None,
        'examples': examples,
        'disclaimer': DISCLAIMER,
    }
    out = Path(args.out)
    save_json(summary, out / 'counterfactual_summary.json')
    save_plot(single_rate, combined_freq, len(ok) / n, out / 'counterfactual.png')
    logger.info("여러 칸으로 승인 가능 %.1f%% (평균 %.2f칸), 한 칸만으로 가능 %.1f%%",
                len(ok) / n * 100, summary['combined']['mean_changed_features'] or 0, any_single * 100)
    for f in feats:
        logger.info("  %-38s 한 칸 %.2f / 함께 %.2f / 중앙 변화 %s", f, single_rate[f], combined_freq[f], med_change(f))
    for e in examples:
        logger.info("  예시 %s: %.3f → %.3f | %s", e['customer_id'], e['probability'], e['new_probability'], ' / '.join(e['changes']))


if __name__ == '__main__':
    main()
