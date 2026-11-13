"""
보너스 과제 3: Two-way ANOVA (과제 노션 「5. 보너스 과제 (선택)」 3번)

고객 세그먼트(씬파일러/일반) × 모델 구성(전통/통합) Two-way ANOVA 와 교호작용 (src/statistics/two_way_anova.py).
  - 데이터: Train + Validation (Test 는 쓰지 않는다, 6.4 교차검증 씬파일러 향상과 같은 방식)
  - 모델: 두 구성 모두 Validation AUC 최고 후보인 LightGBM is_unbalance (6.4와 같음).
    저장된 통합 모델의 설정(하이퍼파라미터)만 가져와, 폴드마다 전통 칸만 / 통합 칸으로 다시 학습한다.
  - 관측값: 5-Fold × 2회 반복 = 칸마다 10개 폴드 AUC
  - 결과: outputs/statistics/two_way_anova.json, two_way_anova.png

사용법
  python main_bonus_two_way_anova.py
  python main_bonus_two_way_anova.py --repeats 3
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
MODELS = PROJECT_ROOT / 'models'
DROP = ['SeriousDlqin2yrs', 'gender', 'age_group', 'is_thin_filer', 'customer_id']


def parse_args():
    p = argparse.ArgumentParser(description='보너스 3: Two-way ANOVA (세그먼트 × 모델 구성)')
    p.add_argument('--splits', default=os.getenv('SPLITS_DIR', str(PROJECT_ROOT / 'data' / 'splits')))
    p.add_argument('--model', default=os.getenv('MODEL_PATH', str(MODELS / 'lightgbm_is_unbalance_integrated_v1.0.joblib')),
                   help='설정을 가져올 모델 (두 구성 모두 이 설정으로 폴드마다 다시 학습)')
    p.add_argument('--traditional-model', help='전통 구성에 다른 설정을 쓸 때만 (없으면 --model)')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--repeats', type=int, default=int(os.getenv('TWO_WAY_REPEATS', '2')))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default=os.getenv('ANOVA_OUTPUT_DIR', str(PROJECT_ROOT / 'outputs' / 'statistics')))
    return p.parse_args()


def save_plot(anova, paired, path):
    import numpy as np
    from src.explainer import shap_explainer as sx
    from src.statistics.two_way_anova import CONFIGS, SEGMENTS
    plt = sx._setup_font()
    cm, sd, n = anova['cell_means'], anova['cell_std'], anova['n_per_cell']
    colors = {'씬파일러': '#D95B5B', '일반': '#5B6CD9'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(CONFIGS))
    for seg in SEGMENTS:
        m = [cm[f'{seg}|{c}'] for c in CONFIGS]
        e = [1.96 * sd[f'{seg}|{c}'] / np.sqrt(n) for c in CONFIGS]
        ax1.errorbar(x, m, yerr=e, marker='o', capsize=4, lw=2, color=colors[seg], label=seg)
        for xi, mi in zip(x, m):
            ax1.annotate(f'{mi:.3f}', (xi, mi), textcoords='offset points', xytext=(10, -4), fontsize=9, color=colors[seg])
    ax1.set_xticks(x, [f'{c} 모델' for c in CONFIGS])
    ax1.set_xlim(-0.4, 1.6)
    ax1.set_ylabel('폴드 AUC 평균 (95% 구간)')
    inter = next(r for r in anova['table'] if r['source'] == 'segment×config')
    ax1.set_title(f"교호작용 그림 (교호작용 F = {inter['f_statistic']:.1f}, p = {inter['p_value']:.2g})")
    ax1.legend(title='고객 세그먼트')

    folds = paired['per_fold']
    idx = np.arange(len(folds))
    w = 0.38
    for i, seg in enumerate(SEGMENTS):
        ax2.bar(idx + (i - 0.5) * w, [f[f'uplift_{seg}'] for f in folds], w, color=colors[seg], label=seg)
    ax2.axhline(0, color='grey', lw=0.8)
    ax2.set_xticks(idx, [str(f['fold_id'] + 1) for f in folds])
    ax2.set_xlabel('폴드')
    ax2.set_ylabel('AUC 향상 (통합 - 전통)')
    ax2.set_title('폴드마다 대안 데이터를 넣었을 때의 AUC 향상')
    ax2.legend(title='고객 세그먼트')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    import joblib
    import numpy as np
    import pandas as pd

    from src.explainer.performance import DISCLAIMER, save_json
    from src.models.train import get_features_by_type
    from src.statistics import two_way_anova as tw

    splits = Path(args.splits)
    dev = pd.concat([pd.read_csv(splits / 'train_processed.csv'), pd.read_csv(splits / 'val_processed.csv')],
                    ignore_index=True)
    y = dev['SeriousDlqin2yrs'].to_numpy()
    thin = dev['is_thin_filer'].to_numpy().astype(bool)
    X = dev.drop(columns=[c for c in DROP if c in dev.columns])
    logger.info("Train+Validation %d명 (씬파일러 %d명, 그중 부도 %d명)", len(dev), thin.sum(), y[thin].sum())

    base = joblib.load(args.model)
    templates = {'전통': joblib.load(args.traditional_model) if args.traditional_model else base, '통합': base}
    records = tw.fold_auc_records(X, y, thin, templates, get_features_by_type,
                                  n_splits=args.folds, n_repeats=args.repeats, seed=args.seed)
    anova = tw.two_way_anova(records)
    paired = tw.paired_checks(records)
    sentences = tw.interpret(anova, paired)
    for s in sentences:
        logger.info(s)

    out = Path(args.out)
    result = {
        'test_name': '고객 세그먼트(씬파일러/일반) × 모델 구성(전통/통합) Two-way ANOVA',
        'response': 'fold AUC',
        'design': {'data': 'Train + Validation (Test 미사용)', 'cv': f'{args.folds}-fold × {args.repeats}회 반복 (층: 부도 × 씬파일러)',
                   'seed': args.seed, 'traditional_model': Path(args.traditional_model or args.model).name,
                   'integrated_model': Path(args.model).name,
                   'n_customers': int(len(dev)), 'n_thin_filer': int(thin.sum()), 'n_thin_filer_default': int(y[thin].sum())},
        'anova': anova,
        'paired_checks': paired,
        'interpretation': sentences,
        'records': records,
        'disclaimer': DISCLAIMER,
    }
    save_json(result, out / 'two_way_anova.json')
    save_plot(anova, paired, out / 'two_way_anova.png')
    logger.info("저장: %s, %s", out / 'two_way_anova.json', out / 'two_way_anova.png')


if __name__ == '__main__':
    main()
