"""학습 결과 기록 테스트: 모델별 KS(#28), PSI(#28), 씬파일러 AUC 향상(#27)"""
import json

import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip('mlflow')
if not hasattr(mlflow, 'start_run'):  # 저장소의 mlflow/ 폴더(서버 설정)만 잡히고 패키지는 없는 경우
    pytest.skip('mlflow 패키지가 설치되어 있지 않음 (pip install -r requirements.txt)', allow_module_level=True)
pytest.importorskip('lightgbm')
pytest.importorskip('imblearn')

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402

from src.models import train as tr  # noqa: E402

FIN = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio']
ALT = ['telecom_payment_rate', 'utility_payment_rate']


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    X = pd.DataFrame({
        'RevolvingUtilizationOfUnsecuredLines': rng.normal(size=n) + 0.3 * y,
        'age': rng.normal(size=n),
        'DebtRatio': rng.normal(size=n),
        'telecom_payment_rate': rng.normal(size=n) - 1.5 * y,
        'utility_payment_rate': rng.normal(size=n) - 1.0 * y,
    })
    return X, pd.Series(y)


def test_cv_scoring_records_ks():
    X, y = _data()
    scores = cross_validate(LogisticRegression(max_iter=500), X, y, scoring=tr.CV_SCORING, return_train_score=True,
                            cv=StratifiedKFold(5, shuffle=True, random_state=0))
    assert len(scores['test_ks']) == 5
    assert all(0 <= v <= 1 for v in scores['test_ks'])
    means = tr._log_cv_results(scores, method='test')
    assert 'cv_test_ks_mean' in means


def test_select_and_test_once_records_psi_and_uplift(tmp_path, monkeypatch):
    monkeypatch.setenv('MLFLOW_TRACKING_URI', tmp_path.joinpath('mlruns').as_uri())
    monkeypatch.setenv('UPLIFT_BOOTSTRAP', '100')
    X_tr, y_tr = _data(seed=1)
    X_v, y_v = _data(seed=2)
    X_te, y_te = _data(seed=3)
    thin = pd.Series([1] * 300 + [0] * 300)
    results = {}
    for data_type in ('financial_only', 'alternative_only', 'integrated'):
        cols = tr.get_features_by_type(X_tr, data_type).columns
        model = LogisticRegression(max_iter=500).fit(X_tr[cols], y_tr)
        results[data_type] = {'LogisticRegression': {'class_weight': model}}
    out = tr.select_and_test_once(results, X_v, y_v, X_te, y_te, models_dir=str(tmp_path),
                                  X_train=X_tr, thin_mask=thin)
    data_type = out['selected_model'].split('/')[0]
    assert set(out['psi']['feature_psi']['per_feature']) == set(tr.get_features_by_type(X_te, data_type).columns)
    assert out['psi']['score_psi'] >= 0
    u = out['thin_filer_uplift']
    assert u['model_without_alt'].startswith('financial_only/')
    assert u['model_with_alt'].startswith('integrated/')
    assert u['n_thin_filer'] == 300
    assert u['ci_low'] <= u['uplift'] <= u['ci_high']
    meta = json.loads((tmp_path / 'model_metadata.json').read_text(encoding='utf-8'))
    assert 'psi' in meta['selection'] and 'thin_filer_uplift' in meta['selection']
