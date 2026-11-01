"""main_step3가 model_metadata.json의 교차검증 씬파일러 향상을 가져오는지 (#26)"""
import json

from main_step3 import cv_uplift_from_metadata

CV = {'n_thin_filer': 3485, 'n_thin_filer_default': 253, 'auc_without_alt': 0.749, 'auc_with_alt': 0.815,
      'uplift': 0.066, 'ci_low': 0.042, 'ci_high': 0.089, 'target': 0.03, 'pass': True}


def _write(tmp_path, selection):
    (tmp_path / 'model_metadata.json').write_text(json.dumps({'selection': selection}), encoding='utf-8')


def test_reads_cv_uplift_for_selected_model(tmp_path):
    _write(tmp_path, {'selected_model': 'integrated/LightGBM/is_unbalance', 'thin_filer_uplift_cv': CV})
    assert cv_uplift_from_metadata(tmp_path / 'lightgbm_is_unbalance_integrated_v1.0.joblib') == CV


def test_matches_underscored_algorithm_name(tmp_path):
    _write(tmp_path, {'selected_model': 'integrated/LogisticRegression/class_weight', 'thin_filer_uplift_cv': CV})
    assert cv_uplift_from_metadata(tmp_path / 'logistic_regression_class_weight_integrated_v1.0.joblib') == CV


def test_other_model_gets_none(tmp_path):
    # 최종 모델이 아닌 모델을 평가할 때는 최종 모델의 교차검증 값을 붙이지 않는다
    _write(tmp_path, {'selected_model': 'integrated/LightGBM/is_unbalance', 'thin_filer_uplift_cv': CV})
    assert cv_uplift_from_metadata(tmp_path / 'lightgbm_is_unbalance_financial_only_v1.0.joblib') is None


def test_missing_metadata_or_value_gets_none(tmp_path):
    assert cv_uplift_from_metadata(tmp_path / 'lightgbm_is_unbalance_integrated_v1.0.joblib') is None
    _write(tmp_path, {'selected_model': 'integrated/LightGBM/is_unbalance'})
    assert cv_uplift_from_metadata(tmp_path / 'lightgbm_is_unbalance_integrated_v1.0.joblib') is None
