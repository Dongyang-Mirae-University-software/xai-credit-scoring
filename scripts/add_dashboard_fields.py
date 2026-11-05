"""
대시보드 시안용 추가 데이터를 만든다 (이슈 #116). 학습·모델 파일은 건드리지 않는다.

  data/customers.json            GET /customers/{id} 재료 (원본 GMSC + 대안 데이터 시뮬레이터 재현, 1000명)
  shap_summary.json              + beeswarm, features[].higher_is_riskier
  metrics_<모델>.json            + baseline (Logistic Regression, 같은 Test 에서 계산)
  fairness*.json                 + evaluated_at
  anova.json                     + source.cv

실행: python scripts/add_dashboard_fields.py   (data/raw/cs-training.csv 필요, 약 2분)
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.disable(logging.INFO)

SHAP_SAMPLE, BEESWARM_N, N_CUSTOMERS = 2000, 60, 1000
CV = {'n_splits': 5, 'n_repeats': 1}   # src/models/train.py 의 StratifiedKFold(n_splits=5), 반복 없음
GENDER = {0: '여성', 1: '남성'}
PERCENT = ('DebtRatio', 'telecom_payment_rate', 'utility_payment_rate')


def patch(path, update):
    d = json.loads(path.read_text(encoding='utf-8'))
    update(d)
    path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')


def customers(raw, train_ids):
    """원래 값 → /predict 입력 단위(부채비율·납부율 %). 전처리와 같게 월소득 결측 행의 DebtRatio 는 중앙값으로 나눈다."""
    from src.api.schemas import CustomerInput
    df = raw.copy()
    median = df.loc[df['Unnamed: 0'].isin(train_ids), 'MonthlyIncome'].median()
    miss = df['MonthlyIncome'].isna()
    df.loc[miss, 'DebtRatio'] = df.loc[miss, 'DebtRatio'] / median
    df = df[~miss & (df['RevolvingUtilizationOfUnsecuredLines'] <= 2) & (df['DebtRatio'] <= 5)
            & df['NumberOfDependents'].notna() & df['age'].between(18, 120)]
    df = df[df['Unnamed: 0'] != 1].sample(N_CUSTOMERS - 1, random_state=42)
    df = df._append(raw[raw['Unnamed: 0'] == 1])   # 요청서 예시 고객 C100001
    out = []
    fields = [f.alias or n for n, f in CustomerInput.model_fields.items() if n != 'customer_id']
    for _, r in df.iterrows():
        row = {'customer_id': f"C{100000 + int(r['Unnamed: 0'])}", 'gender': GENDER[int(r['gender'])]}
        for k in fields:
            v = float(r[k]) * (100 if k in PERCENT else 1)
            row[k] = round(v, 4) if k in PERCENT or k.endswith('Lines') or k == 'spending_consistency' else int(round(v))
        row['NumberOfDependents'] = int(r['NumberOfDependents'])
        out.append(CustomerInput(**{**row}).model_dump(by_alias=True) | {'gender': row['gender']})  # 범위 검증
    return out


def main():
    import joblib
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression

    from src.data.loader import load_gmsc
    from src.data.preprocessor import get_train_index, preprocess, split_data
    from src.data.simulator import generate_alternative_data
    from src.explainer import performance as perf
    from src.explainer import shap_explainer as sx

    raw = load_gmsc(str(ROOT / 'data/raw/cs-training.csv'))
    raw = generate_alternative_data(raw, target_col='SeriousDlqin2yrs', thin_filer_ratio=None, bias_ratio=0, random_seed=42)
    train_idx = get_train_index(raw)
    processed, _ = preprocess(raw, fit_index=train_idx)
    split_data(processed)                      # data/splits/ (gitignored) 재현
    sp = ROOT / 'data/splits'
    train, test = pd.read_csv(sp / 'train_processed.csv'), pd.read_csv(sp / 'test_processed.csv')

    model_path = next((ROOT / 'models').glob('lightgbm_is_unbalance_integrated_v1.0.joblib'))
    model = joblib.load(model_path)
    features = list(model.feature_names_in_)
    out = ROOT / 'outputs'
    metrics_path = out / f'evaluation/metrics_{model_path.stem}.json'
    # 재현 확인: 저장된 Test AUC 와 같아야 한다
    auc = perf.roc_auc(test['SeriousDlqin2yrs'], model.predict_proba(test[features])[:, 1])
    saved = json.loads(metrics_path.read_text(encoding='utf-8'))['auc']
    assert abs(auc - saved) < 1e-9, f'Test 재현 실패: {auc} != {saved}'

    # 3) beeswarm
    sample = test.sample(n=min(SHAP_SAMPLE, len(test)), random_state=42)
    X = sample[features]
    values = sx.shap_values_for_default(sx.make_explainer(model, train[features]), X)
    pick = np.random.RandomState(42).choice(len(X), BEESWARM_N, replace=False)
    beeswarm, risky = {}, {}
    for j, f in enumerate(features):
        col = X[f].to_numpy()
        lo, hi = np.percentile(col, [1, 99])
        norm = np.clip((col - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros(len(col))
        beeswarm[f] = {'shap': [round(float(v), 4) for v in values[pick, j]],
                       'value': [round(float(v), 4) for v in norm[pick]]}
        risky[f] = bool(spearmanr(col, values[:, j])[0] > 0)   # 값이 클수록 SHAP(부도 쪽)도 큰가

    def add_shap(d):
        d['beeswarm'] = beeswarm
        for item in d['features']:
            item['higher_is_riskier'] = risky[item['feature']]
    patch(out / 'evaluation/shap_summary.json', add_shap)

    # 4) 기준 모델: 같은 Train 으로 학습, 같은 Test 로 평가
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42).fit(train[features], train['SeriousDlqin2yrs'])
    ev = perf.evaluate(test['SeriousDlqin2yrs'], lr.predict_proba(test[features])[:, 1])
    baseline = {'model': 'LogisticRegression', 'auc': ev['auc'], 'roc_curve': ev['roc_curve'], 'pr_curve': ev['pr_curve']}
    patch(metrics_path, lambda d: d.update(baseline=baseline))

    # 5, 6) 점검 시각 · CV 설정
    now = datetime.now().astimezone().isoformat(timespec='seconds')
    for name in ('fairness', 'fairness_mitigation'):
        patch(out / f'fairness/{name}.json', lambda d: d.update(evaluated_at=now))
    patch(out / 'statistics/anova.json', lambda d: d.setdefault('source', {}).update(cv=CV))

    # 2) 고객
    custs = customers(raw, set(raw.loc[train_idx, 'Unnamed: 0']))
    (ROOT / 'data/customers.json').write_text(json.dumps(custs, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(f"완료: 고객 {len(custs)}명, 기준 모델 AUC {baseline['auc']:.4f}")


if __name__ == '__main__':
    main()
