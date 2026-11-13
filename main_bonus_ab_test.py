"""
보너스 과제 5: A/B 테스트 시뮬레이션 (과제 노션 「5. 보너스 과제 (선택)」 5번)

A = 기존 CB 모델 (금융 칸 10개만), B = 대안신용평가 모델 (금융 10개 + 대안 5개, 지금 서비스 모델).
  - A 는 B 와 같은 설정(LightGBM is_unbalance)으로 Train 의 금융 칸만 써서 학습한다.
  - Test 22,500명에서 정책 3개(같은 기준값 / 같은 승인율 / 같은 대손율)를 비교하고,
    고객을 무작위로 반씩 나누는 A/B 테스트를 1회 + 1,000회 반복한다 (src/statistics/ab_test.py).
  - 결과: outputs/evaluation/ab_test/ab_test.json, ab_test.png

사용법
  python main_bonus_ab_test.py
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
MODELS = PROJECT_ROOT / 'models'
POLICY_KR = {'same_threshold': '같은 기준값', 'same_approval_rate': '같은 승인율', 'same_bad_rate': '같은 대손율'}


def parse_args():
    p = argparse.ArgumentParser(description='보너스 5: A/B 테스트 시뮬레이션 (기존 CB 모델 vs 대안신용평가 모델)')
    p.add_argument('--model', default=os.getenv('MODEL_PATH', str(MODELS / 'lightgbm_is_unbalance_integrated_v1.0.joblib')),
                   help='B 모델 (A 는 이 설정으로 금융 칸만 다시 학습)')
    p.add_argument('--splits', default=os.getenv('SPLITS_DIR', str(PROJECT_ROOT / 'data' / 'splits')))
    p.add_argument('--out', default=os.getenv('AB_OUTPUT_DIR', str(PROJECT_ROOT / 'outputs' / 'evaluation' / 'ab_test')))
    p.add_argument('--repeats', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def save_plot(y, p_a, p_b, thin, result, path):
    import numpy as np
    from src.explainer import shap_explainer as sx
    from src.statistics import ab_test as ab
    plt = sx._setup_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.35, 1]})
    styles = [('A 기존 CB 모델', p_a, '#8C8C8C'), ('B 대안신용평가 모델', p_b, '#2A7FBF')]
    for label, p, c in styles:
        cur = ab.strategy_curve(y, p)
        ax1.plot(np.array(cur['approval_rate']) * 100, np.array(cur['bad_rate']) * 100, color=c, lw=2, label=f'{label} (전체)')
        cur = ab.strategy_curve(y, p, mask=thin)
        ax1.plot(np.array(cur['approval_rate']) * 100, np.array(cur['bad_rate']) * 100, color=c, lw=1.4, ls='--',
                 label=f'{label} (씬파일러)')
    pol = result['policies']['policies']
    a = result['policies']['a']
    ax1.scatter([a['approval_rate'] * 100], [a['bad_rate'] * 100], color='#555', zorder=5, s=40)
    ax1.annotate('A 기준값 0.5', (a['approval_rate'] * 100, a['bad_rate'] * 100), textcoords='offset points',
                 xytext=(-80, 8), fontsize=9)
    for key, dx, dy in (('same_approval_rate', -78, -16), ('same_bad_rate', 6, 8), ('same_threshold', 4, -18)):
        b = pol[key]['b']
        ax1.scatter([b['approval_rate'] * 100], [b['bad_rate'] * 100], color='#2A7FBF', zorder=5, s=40)
        ax1.annotate(f"B {POLICY_KR[key]}", (b['approval_rate'] * 100, b['bad_rate'] * 100), textcoords='offset points',
                     xytext=(dx, dy), fontsize=9, color='#1F5F8F')
    ax1.set_xlabel('승인율 (%)')
    ax1.set_ylabel('대손율: 승인 고객 중 부도 (%)')
    ax1.set_xlim(50, 100)
    ax1.set_ylim(0, None)
    ax1.set_title('승인율과 대손율 (아래·오른쪽일수록 좋음)')
    ax1.legend(fontsize=8, loc='upper left')

    rep = result['ab_repeated']
    d = np.array(rep['bad_rate_diffs']) * 100
    ax2.hist(d, bins=40, color='#2A7FBF', alpha=0.8)
    ax2.axvline(0, color='grey', lw=0.8)
    ax2.axvline(d.mean(), color='#1F5F8F', ls='--', lw=1.2)
    ax2.set_xlabel('대손율 차이 B - A (%p)')
    ax2.set_ylabel('반복 횟수')
    ax2.set_title(f"A/B 테스트 {rep['n_rep']:,}회 반복 (같은 승인율 정책)\n"
                  f"평균 {d.mean():+.2f}%p, 유의하게 나온 비율 {rep['share_significant_bad_rate']:.0%}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    import joblib
    import pandas as pd
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score

    from src.explainer.performance import DISCLAIMER, save_json
    from src.models.train import get_features_by_type
    from src.statistics import ab_test as ab

    splits = Path(args.splits)
    train = pd.read_csv(splits / 'train_processed.csv')
    test = pd.read_csv(splits / 'test_processed.csv')
    y_tr, y = train['SeriousDlqin2yrs'].to_numpy(), test['SeriousDlqin2yrs'].to_numpy()
    thin = test['is_thin_filer'].to_numpy().astype(bool)
    threshold = float(os.getenv('APPROVAL_THRESHOLD', '0.5'))

    model_b = joblib.load(args.model)
    cols_b = list(getattr(model_b, 'feature_names_in_', get_features_by_type(test, 'integrated').columns))
    p_b = model_b.predict_proba(test[cols_b])[:, 1]
    X_a_tr = get_features_by_type(train, 'financial_only')
    model_a = clone(model_b).fit(X_a_tr, y_tr)
    p_a = model_a.predict_proba(get_features_by_type(test, 'financial_only'))[:, 1]
    logger.info("Test AUC: A 기존 CB %.3f, B 대안신용평가 %.3f", roc_auc_score(y, p_a), roc_auc_score(y, p_b))

    policies = ab.policy_comparison(y, p_a, p_b, threshold, segment=thin)
    for key, row in policies['policies'].items():
        a, b = policies['a'], row['b']
        logger.info("%s: 승인율 A %.1f%% → B %.1f%%, 대손율 A %.2f%% → B %.2f%% (상대 %+.1f%%)", POLICY_KR[key],
                    a['approval_rate'] * 100, b['approval_rate'] * 100, a['bad_rate'] * 100, b['bad_rate'] * 100,
                    row['bad_rate_relative_change'] * 100)
    thr_b = policies['policies']['same_approval_rate']['threshold_b']
    single = ab.ab_split(y, p_a, p_b, threshold, thr_b, seed=args.seed)
    repeated = ab.repeated_ab(y, p_a, p_b, threshold, thr_b, n_rep=args.repeats, seed=args.seed)
    logger.info("A/B 1회 (같은 승인율): 대손율 차이 %+.2f%%p, p = %.3g / %d회 반복 중 유의 %.0f%%",
                single['bad_rate_test']['diff'] * 100, single['bad_rate_test']['p_value'], args.repeats,
                repeated['share_significant_bad_rate'] * 100)
    same = policies['policies']['same_approval_rate']
    needed = ab.required_sample_size(policies['a']['bad_rate'], same['b']['bad_rate'], policies['a']['approval_rate'])
    logger.info("이 차이를 검정력 80%%로 찾으려면 한쪽 신청 고객 약 %d명 (모두 %d명) 필요", needed['applicants_per_arm'],
                needed['applicants_total'])

    result = {
        'models': {'a': '기존 CB 모델: 금융 칸 10개, LightGBM is_unbalance (Train 으로 학습)',
                   'b': f'대안신용평가 모델: {Path(args.model).name}'},
        'test_auc': {'a': float(roc_auc_score(y, p_a)), 'b': float(roc_auc_score(y, p_b))},
        'bad_rate_definition': '승인 고객 중 부도(SeriousDlqin2yrs = 1) 비율 (금액 정보가 없어 건수 기준)',
        'policies': policies,
        'ab_single': {'seed': args.seed, 'policy': 'same_approval_rate', **single},
        'ab_repeated': {'policy': 'same_approval_rate', **repeated},
        'required_sample_size': needed,
        'disclaimer': DISCLAIMER,
    }
    out = Path(args.out)
    save_plot(y, p_a, p_b, thin, result, out / 'ab_test.png')
    result['ab_repeated'].pop('bad_rate_diffs')
    save_json(result, out / 'ab_test.json')
    logger.info("저장: %s, %s", out / 'ab_test.json', out / 'ab_test.png')


if __name__ == '__main__':
    main()
