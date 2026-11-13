"""
보너스 과제 4: 모델 드리프트 탐지 (과제 노션 「5. 보너스 과제 (선택)」 4번)

Train 을 기준으로, Test 와 Test 를 일부러 바꾼 시나리오 3개에서 PSI(점수) 와 CSI(칸별) 를 잰다 (src/statistics/drift.py).
  0) 기준 확인: Train vs Test 그대로
  1) 경기 악화: 한도 사용 비율 +15%p, 부채 비율 ×1.3, 고객 20%가 30~59일 연체 1회 추가
  2) 씬파일러 유입 증가: Test 를 다시 뽑아 씬파일러 비율을 약 3% → 15% 로
  3) 대안 데이터 집계 변경: 앱 로그인 빈도 집계 기준이 바뀌어 값이 절반으로 들어옴
각 시나리오에서 점수 PSI, 칸별 CSI, AUC, 승인율을 함께 기록한다.
  - 결과: outputs/monitoring/drift_report.json, drift_csi.png

사용법
  python main_bonus_drift.py
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
MODELS = PROJECT_ROOT / 'models'

SHORT = {'baseline': '기준\n(Test 그대로)', 'recession': '경기\n악화', 'thin_influx': '씬파일러\n유입 증가',
         'app_login_change': '앱 로그인\n집계 변경'}
SCENARIOS = {
    'baseline': '0) 기준 확인 (Test 그대로)',
    'recession': '1) 경기 악화',
    'thin_influx': '2) 씬파일러 유입 증가',
    'app_login_change': '3) 앱 로그인 집계 변경',
}


def parse_args():
    p = argparse.ArgumentParser(description='보너스 4: PSI + CSI 드리프트 탐지')
    p.add_argument('--model', default=os.getenv('MODEL_PATH', str(MODELS / 'lightgbm_is_unbalance_integrated_v1.0.joblib')))
    p.add_argument('--scaler', default=os.getenv('SCALER_PATH', str(MODELS / 'preprocess_scaler_v1.0.joblib')))
    p.add_argument('--splits', default=os.getenv('SPLITS_DIR', str(PROJECT_ROOT / 'data' / 'splits')))
    p.add_argument('--out', default=os.getenv('DRIFT_OUTPUT_DIR', str(PROJECT_ROOT / 'outputs' / 'monitoring')))
    p.add_argument('--thin-share', type=float, default=0.15, help='시나리오 2 의 씬파일러 비율')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def make_scenarios(test, raw, scaler, cols, thin_share, seed):
    """{이름: (처리된 DataFrame, 원래 단위 DataFrame)}. 처리된 값 = scaler 로 다시 바꾼 값."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)

    def rescale(r, base):
        out = base.copy()
        out[cols] = scaler.transform(r[cols])
        return out

    out = {'baseline': (test, raw)}

    r = raw.copy()
    r['RevolvingUtilizationOfUnsecuredLines'] += 0.15
    r['DebtRatio'] *= 1.3
    hit = rng.random(len(r)) < 0.20
    r.loc[hit, 'NumberOfTime30-59DaysPastDueNotWorse'] += 1
    out['recession'] = (rescale(r, test), r)

    thin = test['is_thin_filer'].to_numpy().astype(bool)
    n = len(test)
    n_thin = int(round(n * thin_share))
    idx = np.concatenate([rng.choice(np.where(thin)[0], n_thin, replace=True),
                          rng.choice(np.where(~thin)[0], n - n_thin, replace=False)])
    out['thin_influx'] = (test.iloc[idx].reset_index(drop=True), raw.iloc[idx].reset_index(drop=True))

    r = raw.copy()
    r['app_login_frequency'] = np.floor(r['app_login_frequency'] / 2)
    out['app_login_change'] = (rescale(r, test), r)
    return out


def save_plot(reports, path):
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from src.explainer import shap_explainer as sx
    from src.statistics.drift import CAUTION, STABLE
    plt = sx._setup_font()
    names = list(reports)
    feats = list(reports[names[0]]['csi'])
    mat = np.array([[reports[s]['csi'][f]['index'] for s in names] for f in feats])
    cmap = ListedColormap(['#E8F3EC', '#F6D98B', '#E58A7B'])
    norm = BoundaryNorm([0, STABLE, CAUTION, 1e9], cmap.N)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.4), gridspec_kw={'width_ratios': [1.5, 1]})
    ax1.imshow(mat, cmap=cmap, norm=norm, aspect='auto')
    ax1.set_xticks(range(len(names)), [SHORT[s] for s in names], fontsize=9)
    ax1.set_yticks(range(len(feats)), [sx._label(f) for f in feats], fontsize=9)
    for i in range(len(feats)):
        for j in range(len(names)):
            v = mat[i, j]
            ax1.text(j, i, f'{v:.3f}' if v >= 0.001 else '0', ha='center', va='center', fontsize=8)
    ax1.set_title('칸별 CSI (초록 < 0.1 안정, 노랑 0.1 ~ 0.25 주의, 빨강 ≥ 0.25 경고)', fontsize=10)

    x = np.arange(len(names))
    psis = [reports[s]['score_psi'] for s in names]
    colors = ['#8BC4A0' if v < STABLE else '#E8B84A' if v < CAUTION else '#D9614F' for v in psis]
    ax2.bar(x, psis, color=colors)
    ax2.axhline(STABLE, ls='--', color='grey', lw=0.8)
    ax2.text(len(names) - 0.5, STABLE, '안정 기준 0.1', ha='right', va='bottom', fontsize=8, color='grey')
    top = max(max(psis) * 1.6, STABLE * 1.5)
    for xi, s in zip(x, names):
        r = reports[s]
        ax2.text(xi, r['score_psi'] + top * 0.015,
                 f"PSI {r['score_psi']:.3f}\nAUC {r['auc']:.3f}\n승인율 {r['approval_rate']:.1%}",
                 ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(x, [SHORT[s] for s in names], fontsize=9)
    ax2.set_ylim(0, top)
    ax2.set_ylabel('점수 PSI')
    ax2.set_title('모델 점수 PSI와 AUC · 승인율', fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from src.explainer.performance import DISCLAIMER, save_json
    from src.statistics import drift

    model, scaler = joblib.load(args.model), joblib.load(args.scaler)
    cols = list(scaler.feature_names_in_)
    model_cols = list(getattr(model, 'feature_names_in_', cols))
    threshold = float(os.getenv('APPROVAL_THRESHOLD', '0.5'))
    splits = Path(args.splits)
    train = pd.read_csv(splits / 'train_processed.csv')
    test = pd.read_csv(splits / 'test_processed.csv')

    def to_raw(df):
        return pd.DataFrame(scaler.inverse_transform(df[cols]), columns=cols)

    train_raw = to_raw(train)
    train_scores = model.predict_proba(train[model_cols])[:, 1]
    scenarios = make_scenarios(test, to_raw(test), scaler, cols, args.thin_share, args.seed)

    reports = {}
    for key, (proc, raw) in scenarios.items():
        scores = model.predict_proba(proc[model_cols])[:, 1]
        rep = drift.drift_report(train_raw, raw, cols, train_scores, scores, name=SCENARIOS[key])
        rep['auc'] = float(roc_auc_score(proc['SeriousDlqin2yrs'], scores))
        rep['approval_rate'] = float((scores < threshold).mean())
        rep['thin_filer_share'] = float(proc['is_thin_filer'].mean())
        rep['description'] = SCENARIOS[key]
        reports[key] = rep
        logger.info("%s: 점수 PSI %.4f (%s), CSI 주의 이상 %s, AUC %.3f, 승인율 %.1f%%", SCENARIOS[key], rep['score_psi'],
                    rep['score_grade'], rep['flagged'] or '없음', rep['auc'], rep['approval_rate'] * 100)

    out = Path(args.out)
    result = {
        'method': 'PSI(점수) + CSI(칸별), 기준 = Train, 구간 10개 (값 종류 10개 이하 칸은 값마다 한 구간)',
        'thresholds': {'stable': drift.STABLE, 'caution': drift.CAUTION},
        'model': Path(args.model).name,
        'scenarios': reports,
        'disclaimer': DISCLAIMER,
    }
    save_json(result, out / 'drift_report.json')
    save_plot(reports, out / 'drift_csi.png')
    logger.info("저장: %s, %s", out / 'drift_report.json', out / 'drift_csi.png')


if __name__ == '__main__':
    main()
