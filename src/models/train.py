"""
모델 학습: Logistic Regression, XGBoost, LightGBM
5-Fold Stratified Cross-Validation + 불균형 처리 (class_weight, SMOTE)
"""

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import mlflow
from mlflow import log_metric, log_param, log_artifact

# MLflow HTTP 요청에 Basic Auth 자동 추가
def _patch_mlflow_auth():
    """MLflow의 requests 세션에 Basic Auth를 자동 추가"""
    from requests.auth import HTTPBasicAuth
    import mlflow.utils.rest_utils as rest_utils

    original_call_endpoint = rest_utils.call_endpoint

    def patched_call_endpoint(host_creds, *args, **kwargs):
        if host_creds and host_creds.password:
            auth = HTTPBasicAuth(host_creds.username, host_creds.password)
            kwargs['auth'] = auth
        return original_call_endpoint(host_creds, *args, **kwargs)

    rest_utils.call_endpoint = patched_call_endpoint

_patch_mlflow_auth()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


def _setup_logger():
    """로깅 설정"""
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def train_models(X_train, y_train, models_dir='models', version='1.0'):
    """
    3가지 알고리즘으로 모델 학습 (5-Fold Stratified CV)
    MLFLOW_TRACKING_URI 환경변수 설정 시 자동으로 MLflow에 기록

    Args:
        X_train: 훈련 데이터 특성
        y_train: 훈련 데이터 타겟
        models_dir: 모델 저장 디렉토리
        version: 모델 버전

    Returns:
        학습된 모델들의 딕셔너리
    """
    import os
    _setup_logger()

    # MLflow 자동 설정 (환경변수 기반)
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow_experiment = os.getenv('MLFLOW_EXPERIMENT_NAME', 'credit-scoring-models')
        mlflow.set_experiment(mlflow_experiment)
        logger.info(f"MLflow 활성화: {mlflow_uri}\n")

    # 모델 저장 디렉토리 생성
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"모델 학습 시작")
    logger.info(f"훈련 데이터: {X_train.shape[0]:,} 샘플, {X_train.shape[1]} 특성")
    logger.info(f"클래스 분포: {(y_train == 0).sum():,} (0), {(y_train == 1).sum():,} (1)")
    logger.info(f"버전: {version}\n")

    # 5-Fold Stratified CV 설정
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    # ===== 모델 1: Logistic Regression =====
    logger.info("="*80)
    logger.info("모델 1: Logistic Regression")
    logger.info("="*80)

    lr_models = train_logistic_regression(X_train, y_train, skf, models_dir, version)
    results['LogisticRegression'] = lr_models

    # ===== 모델 2: XGBoost =====
    logger.info("\n" + "="*80)
    logger.info("모델 2: XGBoost")
    logger.info("="*80)

    xgb_models = train_xgboost(X_train, y_train, skf, models_dir, version)
    results['XGBoost'] = xgb_models

    # ===== 모델 3: LightGBM =====
    logger.info("\n" + "="*80)
    logger.info("모델 3: LightGBM")
    logger.info("="*80)

    lgb_models = train_lightgbm(X_train, y_train, skf, models_dir, version)
    results['LightGBM'] = lgb_models

    logger.info("\n" + "="*80)
    logger.info("✅ 모든 모델 학습 완료")
    logger.info("="*80 + "\n")

    return results


def train_logistic_regression(X_train, y_train, skf, models_dir, version, data_type=None):
    """Logistic Regression 학습 (class_weight, SMOTE 비교)"""

    models = {}

    # data_type이 지정된 경우 MLflow run name에 포함
    run_name_suffix = f"_{data_type}" if data_type else ""

    # 방식 1: class_weight='balanced'
    logger.info("\n[방식 1] class_weight='balanced'")
    logger.info("-" * 80)

    with mlflow.start_run(run_name=f"LR_class_weight{run_name_suffix}"):
        mlflow.log_param("algorithm", "Logistic Regression")
        mlflow.log_param("method", "class_weight")
        if data_type:
            mlflow.log_param("data_type", data_type)
        mlflow.log_param("max_iter", 1000)

        lr_balanced = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ))
        ])

        scores_balanced = cross_validate(
            lr_balanced, X_train, y_train,
            cv=skf,
            scoring=['roc_auc', 'precision', 'recall', 'f1'],
            return_train_score=True,
            n_jobs=-1
        )

        _log_cv_results(scores_balanced, method="class_weight='balanced'")
        lr_balanced.fit(X_train, y_train)
        models['class_weight'] = lr_balanced

        # 모델 저장
        model_path = f"{models_dir}/logistic_regression_class_weight_v{version}.joblib"
        joblib.dump(lr_balanced, model_path)
        logger.info(f"모델 저장: {model_path}\n")

        mlflow.log_artifact(model_path)

    # 방식 2: SMOTE
    logger.info("[방식 2] SMOTE")
    logger.info("-" * 80)

    with mlflow.start_run(run_name=f"LR_SMOTE{run_name_suffix}"):
        mlflow.log_param("algorithm", "Logistic Regression")
        mlflow.log_param("method", "SMOTE")
        if data_type:
            mlflow.log_param("data_type", data_type)
        mlflow.log_param("max_iter", 1000)

        pipeline_smote = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])

        scores_smote = cross_validate(
            pipeline_smote, X_train, y_train,
            cv=skf,
            scoring=['roc_auc', 'precision', 'recall', 'f1'],
            return_train_score=True,
            n_jobs=-1
        )

        _log_cv_results(scores_smote, method="SMOTE")
        pipeline_smote.fit(X_train, y_train)
        models['smote'] = pipeline_smote

        # 모델 저장
        model_path = f"{models_dir}/logistic_regression_smote_v{version}.joblib"
        joblib.dump(pipeline_smote, model_path)
        logger.info(f"모델 저장: {model_path}\n")

        mlflow.log_artifact(model_path)

    return models


def train_xgboost(X_train, y_train, skf, models_dir, version, data_type=None):
    """XGBoost 학습 (scale_pos_weight, SMOTE 비교)"""

    models = {}
    run_name_suffix = f"_{data_type}" if data_type else ""

    # 방식 1: scale_pos_weight (XGBoost의 class_weight 대체)
    logger.info("\n[방식 1] scale_pos_weight")
    logger.info("-" * 80)

    with mlflow.start_run(run_name=f"XGB_scale_pos_weight{run_name_suffix}"):
        # 클래스 불균형 비율 계산
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        mlflow.log_param("algorithm", "XGBoost")
        mlflow.log_param("method", "scale_pos_weight")
        if data_type:
            mlflow.log_param("data_type", data_type)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 2))

        xgb_balanced = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ))
        ])

        scores_balanced = cross_validate(
            xgb_balanced, X_train, y_train,
            cv=skf,
            scoring=['roc_auc', 'precision', 'recall', 'f1'],
            return_train_score=True,
            n_jobs=-1
        )

        _log_cv_results(scores_balanced, method="scale_pos_weight")
        xgb_balanced.fit(X_train, y_train)
        models['scale_pos_weight'] = xgb_balanced

        # 모델 저장
        model_path = f"{models_dir}/xgboost_scale_pos_weight_v{version}.joblib"
        joblib.dump(xgb_balanced, model_path)
        logger.info(f"모델 저장: {model_path}\n")

        mlflow.log_artifact(model_path)

    # 방식 2: SMOTE
    logger.info("[방식 2] SMOTE")
    logger.info("-" * 80)

    with mlflow.start_run(run_name=f"XGB_SMOTE{run_name_suffix}"):
        mlflow.log_param("algorithm", "XGBoost")
        mlflow.log_param("method", "SMOTE")
        if data_type:
            mlflow.log_param("data_type", data_type)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)

        pipeline_smote = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('clf', xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ))
        ])

        scores_smote = cross_validate(
            pipeline_smote, X_train, y_train,
            cv=skf,
            scoring=['roc_auc', 'precision', 'recall', 'f1'],
            return_train_score=True,
            n_jobs=-1
        )

        _log_cv_results(scores_smote, method="SMOTE")
        pipeline_smote.fit(X_train, y_train)
        models['smote'] = pipeline_smote

        # 모델 저장
        model_path = f"{models_dir}/xgboost_smote_v{version}.joblib"
        joblib.dump(pipeline_smote, model_path)
        logger.info(f"모델 저장: {model_path}\n")

        mlflow.log_artifact(model_path)

    return models


def train_lightgbm(X_train, y_train, skf, models_dir, version, data_type=None):
    """LightGBM 학습 (is_unbalance, SMOTE 비교)"""

    models = {}
    run_name_suffix = f"_{data_type}" if data_type else ""

    # 방식 1: is_unbalance (LightGBM의 불균형 처리)
    logger.info("\n[방식 1] is_unbalance=True")
    logger.info("-" * 80)

    with mlflow.start_run(run_name=f"LGBM_is_unbalance{run_name_suffix}"):
        mlflow.log_param("algorithm", "LightGBM")
        mlflow.log_param("method", "is_unbalance")
        if data_type:
            mlflow.log_param("data_type", data_type)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)

        lgb_balanced = Pipeline([
            ('scaler', StandardScaler()),
            ('lgb', lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                is_unbalance=True,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ))
        ])

        scores_balanced = cross_validate(
            lgb_balanced, X_train, y_train,
            cv=skf,
            scoring=['roc_auc', 'precision', 'recall', 'f1'],
            return_train_score=True,
            n_jobs=-1
        )

        _log_cv_results(scores_balanced, method="is_unbalance=True")
        lgb_balanced.fit(X_train, y_train)
        models['is_unbalance'] = lgb_balanced

        # 모델 저장
        model_path = f"{models_dir}/lightgbm_is_unbalance_v{version}.joblib"
        joblib.dump(lgb_balanced, model_path)
        logger.info(f"모델 저장: {model_path}\n")

        mlflow.log_artifact(model_path)

    # 방식 2: SMOTE
    logger.info("[방식 2] SMOTE")
    logger.info("-" * 80)

    with mlflow.start_run(run_name=f"LGBM_SMOTE{run_name_suffix}"):
        mlflow.log_param("algorithm", "LightGBM")
        mlflow.log_param("method", "SMOTE")
        if data_type:
            mlflow.log_param("data_type", data_type)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)

        pipeline_smote = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('clf', lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ))
        ])

        scores_smote = cross_validate(
            pipeline_smote, X_train, y_train,
            cv=skf,
            scoring=['roc_auc', 'precision', 'recall', 'f1'],
            return_train_score=True,
            n_jobs=-1
        )

        _log_cv_results(scores_smote, method="SMOTE")
        pipeline_smote.fit(X_train, y_train)
        models['smote'] = pipeline_smote

        # 모델 저장
        model_path = f"{models_dir}/lightgbm_smote_v{version}.joblib"
        joblib.dump(pipeline_smote, model_path)
        logger.info(f"모델 저장: {model_path}\n")

        mlflow.log_artifact(model_path)

    return models


def _log_cv_results(scores, method, run_name=None):
    """Cross-Validation 결과 로깅 및 MLflow 기록"""

    metrics = ['roc_auc', 'precision', 'recall', 'f1']

    logger.info(f"\n5-Fold CV 결과 ({method}):")
    logger.info("-" * 80)

    for metric in metrics:
        test_scores = scores[f'test_{metric}']
        train_scores = scores[f'train_{metric}']

        logger.info(f"\n{metric.upper()}:")
        for fold, (train, test) in enumerate(zip(train_scores, test_scores), 1):
            logger.info(f"  Fold {fold}: Train={train:.4f}, Test={test:.4f}")

        mean_score = test_scores.mean()
        std_score = test_scores.std()
        logger.info(f"  평균: Test={mean_score:.4f} ± {std_score:.4f}")

        # MLflow 메트릭 기록
        if mlflow.active_run():
            # 5-Fold 개별 점수 기록
            for fold, score in enumerate(test_scores, 1):
                mlflow.log_metric(f"fold_{fold}_{metric}", score)

            # 평균 및 표준편차 기록
            mlflow.log_metric(f"{metric}_mean", mean_score)
            mlflow.log_metric(f"{metric}_std", std_score)

    logger.info("-" * 80)


def get_features_by_type(X_train, data_type):
    """
    데이터 타입별로 특성 분리

    Args:
        X_train: 훈련 데이터 특성
        data_type: 'financial_only' | 'alternative_only' | 'integrated'

    Returns:
        선택된 특성의 부분 데이터프레임
    """
    # 금융 변수
    financial_cols = [
        'RevolvingUtilizationOfUnsecuredLines', 'age',
        'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
        'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
    ]

    # 대안 변수
    alternative_cols = [
        'telecom_payment_rate', 'utility_payment_rate',
        'spending_consistency', 'regular_payment_count',
        'app_login_frequency'
    ]

    if data_type == 'financial_only':
        return X_train[[col for col in financial_cols if col in X_train.columns]]
    elif data_type == 'alternative_only':
        return X_train[[col for col in alternative_cols if col in X_train.columns]]
    else:  # integrated
        return X_train


def train_all_models_by_data_type(X_train, y_train, models_dir='models', version='1.0'):
    """
    3가지 데이터 조합(금융/대안/통합) × 3가지 알고리즘 = 9개 모델 학습

    Args:
        X_train: 훈련 데이터 특성 (모든 변수 포함)
        y_train: 훈련 데이터 타겟
        models_dir: 모델 저장 디렉토리
        version: 모델 버전

    Returns:
        모든 모델의 결과 딕셔너리
    """
    import os

    _setup_logger()

    # MLflow 설정
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
    mlflow_user = os.getenv('MLFLOW_TRACKING_USERNAME')
    mlflow_pass = os.getenv('MLFLOW_TRACKING_PASSWORD')

    if mlflow_uri:
        # MLflow 클라이언트에 자격증명 직접 설정
        if mlflow_user and mlflow_pass:
            from mlflow.utils.credentials import MlflowHostCreds
            from mlflow.tracking._tracking_service import client as tracking_client

            # host_creds 생성
            host_creds = MlflowHostCreds(
                host=mlflow_uri,
                username=mlflow_user,
                password=mlflow_pass
            )

            # 글로벌 클라이언트 초기화
            tracking_client._get_store().get_host_creds = lambda: host_creds

        mlflow.set_tracking_uri(mlflow_uri)

        try:
            mlflow.set_experiment("xai-credit-scoring")
            display_uri = mlflow_uri.split('@')[-1] if '@' in mlflow_uri else mlflow_uri
            logger.info(f"✅ MLflow 활성화: {display_uri}\n")
        except Exception as e:
            logger.warning(f"⚠️ MLflow 연결 실패: {type(e).__name__}\n")
            logger.warning(f"   모델은 로컬({models_dir})에만 저장됩니다\n")
            # MLflow 연결 실패해도 계속 진행 (로컬 저장만)
            mlflow_uri = None

    # 모델 저장 디렉토리 생성
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"모델 학습 시작 (9개 조합)")
    logger.info(f"훈련 데이터: {X_train.shape[0]:,} 샘플, {X_train.shape[1]} 특성")
    logger.info(f"클래스 분포: {(y_train == 0).sum():,} (0), {(y_train == 1).sum():,} (1)")
    logger.info(f"버전: {version}\n")

    # 5-Fold Stratified CV 설정
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_results = {}
    data_types = ['financial_only', 'alternative_only', 'integrated']

    for data_type in data_types:
        logger.info(f"\n{'='*80}")
        logger.info(f"데이터 타입: {data_type}")
        logger.info(f"{'='*80}")

        # 데이터 분리
        X_train_subset = get_features_by_type(X_train, data_type)
        logger.info(f"선택된 특성: {X_train_subset.shape[1]}개\n")

        # 각 데이터 타입별 3개 알고리즘 학습
        results = {}
        results['LogisticRegression'] = train_logistic_regression(
            X_train_subset, y_train, skf, models_dir, version, data_type
        )
        results['XGBoost'] = train_xgboost(
            X_train_subset, y_train, skf, models_dir, version, data_type
        )
        results['LightGBM'] = train_lightgbm(
            X_train_subset, y_train, skf, models_dir, version, data_type
        )

        all_results[data_type] = results

    logger.info(f"\n{'='*80}")
    logger.info(f"✅ 9개 모델 학습 완료 (금융/대안/통합 × 3 알고리즘)")
    logger.info(f"{'='*80}\n")

    return all_results
