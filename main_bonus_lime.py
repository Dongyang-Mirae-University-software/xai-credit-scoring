"""
보너스 과제 1: LIME 비교 분석 (과제 노션 「5. 보너스 과제 (선택)」 1번)

같은 Test 고객에 대해 SHAP 과 LIME 설명을 만들고, 상위 5개 중요 변수 일치율을 계산한다.
  - 표본: Test 에서 거절(부도 확률 ≥ 승인 기준) · 승인 고객을 반씩 무작위로 뽑는다 (기본 200명, seed 42)
  - 결과: outputs/evaluation/lime/lime_vs_shap.json, lime_vs_shap.png

사용법
  python main_bonus_lime.py --model models/lightgbm_is_unbalance_integrated_v1.0.joblib

Test 는 모델 선택에 쓰지 않는다 (이미 고른 최종 모델의 설명만 비교한다).
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
DEFAULT_SPLITS = PROJECT_ROOT / 'data' / 'splits'


def parse_args():
    p = argparse.ArgumentParser(description='보너스 1: LIME vs SHAP 상위 5개 일치율')
    p.add_argument('--model', required=True, help='학습된 모델 파일 (.joblib / .pkl)')
    p.add_argument('--splits', default=os.getenv('SPLITS_DIR', str(DEFAULT_SPLITS)), help='train/test_processed.csv 폴더')
    p.add_argument('--out', default=os.getenv('LIME_OUTPUT_DIR', str(PROJECT_ROOT / 'outputs' / 'evaluation' / 'lime')))
    p.add_argument('--sample', type=int, default=int(os.getenv('LIME_SAMPLE', '200')), help='비교할 Test 고객 수 (거절·승인 반씩)')
    p.add_argument('--stability', type=int, default=int(os.getenv('LIME_STABILITY', '30')),
                   help='LIME 을 다른 시드로 한 번 더 돌려 안정성을 잴 고객 수')
    p.add_argument('--num-samples', type=int, default=5000, help='LIME 이 고객마다 만드는 주변 표본 수')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def pick_sample(test, proba, threshold, n, seed):
    """거절 · 승인 고객을 반씩 무작위로 뽑는다. 한쪽이 모자라면 있는 만큼만."""
    import pandas as pd

    rejected = test[proba >= threshold]
    approved = test[proba < threshold]
    half = n // 2
    parts = [rejected.sample(n=min(half, len(rejected)), random_state=seed),
             approved.sample(n=min(n - half, len(approved)), random_state=seed)]
    return pd.concat(parts)


def save_plot(summary, freq_shap, freq_lime, path):
    """왼쪽: 겹친 변수 개수 분포 / 오른쪽: 변수별 상위 5개에 든 비율 (SHAP vs LIME)."""
    from src.explainer import shap_explainer as sx

    plt = sx._setup_font()
    label = sx._label
    dist = summary['overlap_count_distribution']
    order = sorted(freq_shap, key=lambda c: (freq_shap[c] + freq_lime[c]), reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [1, 1.6]})
    ks = [int(k) for k in dist]
    ax1.bar(ks, [dist[str(k)] for k in ks], color='#5B6CD9')
    for k in ks:
        ax1.text(k, dist[str(k)], str(dist[str(k)]), ha='center', va='bottom', fontsize=9)
    ax1.set_xticks(ks)
    ax1.set_xlabel('SHAP · LIME 상위 5개 중 겹친 변수 수')
    ax1.set_ylabel('고객 수')
    ax1.set_title(f"평균 일치율 {summary['mean_agreement']:.2f} (n={summary['n']})")

    y = range(len(order))
    ax2.barh([i + 0.2 for i in y], [freq_shap[c] for c in order], height=0.4, color='#2A9D8F', label='SHAP')
    ax2.barh([i - 0.2 for i in y], [freq_lime[c] for c in order], height=0.4, color='#E76F51', label='LIME')
    ax2.set_yticks(list(y))
    ax2.set_yticklabels([label(c) for c in order], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('고객별 상위 5개에 든 비율')
    ax2.legend(loc='lower right')
    ax2.set_title('변수별 상위 5개 등장 비율')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    import joblib
    import numpy as np
    import pandas as pd

    from main_step3 import model_features
    from src.explainer import lime_compare as lc
    from src.explainer import shap_explainer as sx
    from src.explainer.performance import DISCLAIMER, save_json

    model_path = Path(args.model)
    model = joblib.load(model_path)
    train = pd.read_csv(Path(args.splits) / 'train_processed.csv')
    test = pd.read_csv(Path(args.splits) / 'test_processed.csv')
    features = model_features(model, list(test.columns))
    threshold = float(os.getenv('APPROVAL_THRESHOLD', '0.5'))

    def predict_proba(x):
        return model.predict_proba(pd.DataFrame(x, columns=features))

    proba = model.predict_proba(test[features])[:, 1]
    sample = pick_sample(test, proba, threshold, args.sample, args.seed)
    X = sample[features]
    logger.info("표본 %d명 (거절 %d, 승인 %d, 씬파일러 %d)", len(X),
                int((model.predict_proba(X)[:, 1] >= threshold).sum()),
                int((model.predict_proba(X)[:, 1] < threshold).sum()),
                int(sample.get('is_thin_filer', pd.Series(0, index=sample.index)).sum()))

    explainer = sx.make_explainer(model, train[features])
    shap_vals = sx.shap_values_for_default(explainer, X)

    lime_exp = lc.make_lime_explainer(train[features], random_state=args.seed)
    lime_vals = np.vstack([lc.lime_contributions(lime_exp, predict_proba, row, len(features), args.num_samples)
                           for row in X.values])

    rows = lc.compare(shap_vals, lime_vals, features)
    p_sample = model.predict_proba(X)[:, 1]
    groups = {
        'all': rows,
        'rejected': [r for r, p in zip(rows, p_sample) if p >= threshold],
        'approved': [r for r, p in zip(rows, p_sample) if p < threshold],
    }
    if 'is_thin_filer' in sample.columns:
        thin = sample['is_thin_filer'].values == 1
        groups['thin_filer'] = [r for r, t in zip(rows, thin) if t]
        groups['thick_filer'] = [r for r, t in zip(rows, thin) if not t]

    # LIME 안정성: 같은 고객을 다른 시드로 한 번 더 설명해 LIME 끼리 비교
    m = min(args.stability, len(X))
    lime_exp2 = lc.make_lime_explainer(train[features], random_state=args.seed + 1)
    lime_vals2 = np.vstack([lc.lime_contributions(lime_exp2, predict_proba, row, len(features), args.num_samples)
                            for row in X.values[:m]])
    stability = lc.summarize(lc.compare(lime_vals[:m], lime_vals2, features))

    freq_shap = lc.top_k_frequency(rows, 'shap_top', features)
    freq_lime = lc.top_k_frequency(rows, 'lime_top', features)
    summary = {name: lc.summarize(g) for name, g in groups.items()}

    result = {
        'task': '보너스 1: LIME 비교 분석 (상위 5개 중요 변수 일치율)',
        'model': model_path.name,
        'data': 'Test (test_processed.csv), 거절 · 승인 반씩 무작위 표본',
        'seed': args.seed,
        'approval_threshold': threshold,
        'top_k': lc.TOP_K,
        'shap': 'TreeExplainer, 부도(1) 쪽 log-odds 기여도',
        'lime': f'LimeTabularExplainer (Train 분포, 구간 나누기), 부도 확률, 주변 표본 {args.num_samples}개',
        'agreement_definition': '고객별 |기여도| 상위 5개 중 겹치는 변수 수 ÷ 5',
        'summary': summary,
        'lime_self_stability': {'n': m, 'seeds': [args.seed, args.seed + 1], **stability},
        'top5_frequency': {'shap': freq_shap, 'lime': freq_lime},
        'examples': [dict(r, customer_id=int(cid)) for r, cid in
                     list(zip(rows, sample.get('customer_id', pd.Series(range(len(rows))))))[:5]],
        'disclaimer': DISCLAIMER,
    }
    out = Path(args.out)
    save_json(result, out / 'lime_vs_shap.json')
    save_plot(summary['all'], freq_shap, freq_lime, out / 'lime_vs_shap.png')

    s = summary['all']
    logger.info("평균 일치율 %.3f (중앙값 %.2f), 1위 일치 %.1f%%, 방향 일치 %.1f%%, LIME 자기 일치율 %.3f",
                s['mean_agreement'], s['median_agreement'], s['top1_same_rate'] * 100,
                (s['mean_sign_agreement'] or 0) * 100, stability['mean_agreement'])
    for name, g in summary.items():
        logger.info("  %-12s n=%3d 평균 일치율 %.3f", name, g['n'], g.get('mean_agreement', float('nan')))


if __name__ == '__main__':
    main()
