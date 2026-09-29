"""
모델 학습 및 평가 (MLflow 통합)

3개 알고리즘:
1. Logistic Regression (베이스라인)
2. XGBoost
3. LightGBM

각 모델:
- 5-Fold Stratified Cross-Validation 적용
- AUC-ROC, KS Statistic, F1 Score 평가
- 최고 성능 모델 선택 및 저장
- MLflow로 실험 추적
"""

import pandas as pd
import numpy as np
import logging
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Tuple, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    f1_score, precision_score, recall_score,
    confusion_matrix
)

logger = logging.getLogger(__name__)

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_SPLITS = PROJECT_ROOT / 'data' / 'splits'
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def load_data():
    """전처리된 데이터 로드"""
    logger.info("데이터 로드 중...")

    def clean_data(df):
        # 인덱스 컬럼 제거
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        # NaN 값 확인 및 제거
        if df.isnull().sum().sum() > 0:
            logger.warning(f"  - NaN 값 발견: {df.isnull().sum().sum()}개 → 제거")
            df = df.dropna()

        return df

    X_train = pd.read_csv(DATA_SPLITS / 'train_processed.csv')
    X_train = clean_data(X_train)
    y_train = X_train['SeriousDlqin2yrs']
    X_train = X_train.drop('SeriousDlqin2yrs', axis=1)

    X_val = pd.read_csv(DATA_SPLITS / 'val_processed.csv')
    X_val = clean_data(X_val)
    y_val = X_val['SeriousDlqin2yrs']
    X_val = X_val.drop('SeriousDlqin2yrs', axis=1)

    X_test = pd.read_csv(DATA_SPLITS / 'test_processed.csv')
    X_test = clean_data(X_test)
    y_test = X_test['SeriousDlqin2yrs']
    X_test = X_test.drop('SeriousDlqin2yrs', axis=1)

    logger.info(f"✓ 데이터 로드 완료")
    logger.info(f"  - 훈련: {X_train.shape}")
    logger.info(f"  - 검증: {X_val.shape}")
    logger.info(f"  - 테스트: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_ks_statistic(y_true, y_pred_proba):
    """
    KS Statistic 계산

    KS = max(|CDF_1 - CDF_0|)
    부도자와 정상자의 누적 분포 최대 차이
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks = np.max(tpr - fpr)
    return ks


def train_model(model, X_train, y_train, model_name: str, cv=None):
    """
    모델 학습 및 5-Fold CV 평가

    Args:
        model: sklearn 모델
        X_train: 훈련 데이터
        y_train: 훈련 타겟
        model_name: 모델 이름
        cv: Cross-Validation 객체

    Returns:
        dict: 검증 점수들
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"🤖 {model_name} 학습 중...")
    logger.info(f"{'='*60}")

    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        'roc_auc': 'roc_auc',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall'
    }

    results = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    # 결과 출력
    logger.info(f"\n5-Fold Cross-Validation 결과:")
    logger.info(f"  AUC-ROC:")
    logger.info(f"    - 훈련: {results['train_roc_auc'].mean():.4f} (±{results['train_roc_auc'].std():.4f})")
    logger.info(f"    - 검증: {results['test_roc_auc'].mean():.4f} (±{results['test_roc_auc'].std():.4f})")
    logger.info(f"    - 각 폴드: {[f'{s:.4f}' for s in results['test_roc_auc']]}")

    logger.info(f"  F1-Score:")
    logger.info(f"    - 훈련: {results['train_f1'].mean():.4f}")
    logger.info(f"    - 검증: {results['test_f1'].mean():.4f}")

    logger.info(f"  Precision:")
    logger.info(f"    - 훈련: {results['train_precision'].mean():.4f}")
    logger.info(f"    - 검증: {results['test_precision'].mean():.4f}")

    logger.info(f"  Recall:")
    logger.info(f"    - 훈련: {results['train_recall'].mean():.4f}")
    logger.info(f"    - 검증: {results['test_recall'].mean():.4f}")

    # 최종 모델 학습 (전체 훈련 데이터)
    model.fit(X_train, y_train)

    return results, model


def evaluate_model(model, X_test, y_test, model_name: str):
    """
    테스트 데이터에서 모델 평가

    Args:
        model: 학습된 모델
        X_test: 테스트 데이터
        y_test: 테스트 타겟
        model_name: 모델 이름

    Returns:
        dict: 평가 메트릭
    """
    logger.info(f"\n테스트 평가:")

    # 예측
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 메트릭 계산
    auc = roc_auc_score(y_test, y_pred_proba)
    ks = calculate_ks_statistic(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    metrics = {
        'auc': auc,
        'ks': ks,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'sensitivity': sensitivity
    }

    logger.info(f"  - AUC-ROC: {auc:.4f}")
    logger.info(f"  - KS Statistic: {ks:.4f}")
    logger.info(f"  - F1-Score: {f1:.4f}")
    logger.info(f"  - Precision: {precision:.4f}")
    logger.info(f"  - Recall (Sensitivity): {recall:.4f}")
    logger.info(f"  - Specificity: {specificity:.4f}")

    return metrics


def main():
    """메인 실행 함수 (MLflow 통합)"""

    logger.info("="*60)
    logger.info("🚀 Step 2: 모델 3중 묶음 학습 시작 (MLflow 추적)")
    logger.info("="*60)

    # MLflow 실험 설정
    experiment_name = "credit_scoring_baseline"
    mlflow.set_experiment(experiment_name)

    # 1. 데이터 로드
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # CV 설정
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 2. 모델 정의
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=1000, solver='lbfgs'
        ),
        'XGBoost': XGBClassifier(
            random_state=42, n_estimators=100, learning_rate=0.1,
            max_depth=6, use_label_encoder=False, eval_metric='logloss'
        ),
        'LightGBM': LGBMClassifier(
            random_state=42, n_estimators=100, learning_rate=0.1,
            num_leaves=31, verbose=-1
        )
    }

    # 3. 각 모델 학습 (MLflow 로깅)
    results_dict = {}
    trained_models = {}

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            logger.info(f"\n📊 {model_name} 실험 시작 (MLflow 로깅 중)...")

            # 하이퍼파라미터 로깅
            params = model.get_params()
            mlflow.log_params({k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))})

            # 모델 학습
            cv_results, trained_model = train_model(model, X_train, y_train, model_name, cv)
            results_dict[model_name] = cv_results
            trained_models[model_name] = trained_model

            # CV 메트릭 로깅
            mlflow.log_metric("cv_auc_mean", cv_results['test_roc_auc'].mean())
            mlflow.log_metric("cv_auc_std", cv_results['test_roc_auc'].std())

    # 4. 테스트 평가
    logger.info(f"\n{'='*60}")
    logger.info("📊 테스트 데이터 평가")
    logger.info(f"{'='*60}")

    test_metrics = {}
    for model_name, model in trained_models.items():
        logger.info(f"\n{model_name}:")
        metrics = evaluate_model(model, X_test, y_test, model_name)
        test_metrics[model_name] = metrics

    # 5. 모델 비교
    logger.info(f"\n{'='*60}")
    logger.info("🏆 모델 성능 비교 (테스트 데이터)")
    logger.info(f"{'='*60}")

    comparison = pd.DataFrame(test_metrics).T
    logger.info(f"\n{comparison.to_string()}")

    # 6. 최고 성능 모델 선택 및 MLflow 등록
    best_model_name = comparison['auc'].idxmax()
    best_auc = comparison.loc[best_model_name, 'auc']
    best_model = trained_models[best_model_name]

    logger.info(f"\n{'='*60}")
    logger.info(f"🌟 최고 성능 모델: {best_model_name}")
    logger.info(f"   Test AUC: {best_auc:.4f}")
    logger.info(f"{'='*60}")

    # MLflow에 최고 성능 모델 정보 로깅
    with mlflow.start_run(run_name=f"{best_model_name}_BEST"):
        mlflow.log_params(best_model.get_params())
        for metric_name, metric_value in test_metrics[best_model_name].items():
            mlflow.log_metric(metric_name, metric_value)
        logger.info(f"✓ MLflow에 최고 성능 모델 메트릭 로깅 완료")

    # 7. 모델 저장
    model_path = MODELS_DIR / f'best_model_{best_model_name.lower().replace(" ", "_")}_v1.0.pkl'
    joblib.dump(best_model, model_path)
    logger.info(f"\n✓ 모델 저장: {model_path}")

    # 8. 메타데이터 저장
    import json
    metadata = {
        'best_model': best_model_name,
        'test_auc': float(best_auc),
        'test_metrics': {k: float(v) for k, v in test_metrics[best_model_name].items()},
        'all_models': {
            model_name: {k: float(v) for k, v in metrics.items()}
            for model_name, metrics in test_metrics.items()
        }
    }

    with open(MODELS_DIR / 'model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ 메타데이터 저장: {MODELS_DIR / 'model_metadata.json'}")

    # 9. 성능 목표 확인
    logger.info(f"\n{'='*60}")
    logger.info("✅ 성능 목표 확인")
    logger.info(f"{'='*60}")
    logger.info(f"  - AUC ≥ 0.78: {'✓' if best_auc >= 0.78 else '✗'} ({best_auc:.4f})")
    logger.info(f"  - KS ≥ 0.28: {'✓' if test_metrics[best_model_name]['ks'] >= 0.28 else '✗'} ({test_metrics[best_model_name]['ks']:.4f})")

    logger.info(f"\n{'='*60}")
    logger.info("✅ Step 2 완료! (MLflow 로깅 완료)")
    logger.info(f"{'='*60}")

    return best_model, best_model_name, test_metrics


if __name__ == '__main__':
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    main()
