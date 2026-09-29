"""
불균형 데이터 처리 기법 비교 실험

비교 대상:
1. No Balancing (기존, class_weight 없음)
2. class_weight='balanced' (가중치 조정)
3. SMOTE (오버샘플링)
4. SMOTE + ENN (결합)

평가: 5-Fold Stratified CV + Test Set
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Dict, Tuple
import mlflow

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline as ImbPipeline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_SPLITS = PROJECT_ROOT / 'data' / 'splits'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'imbalance_comparison'
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def load_data():
    """전처리된 데이터 로드"""
    logger.info("데이터 로드 중...")

    def clean_data(df):
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        if df.isnull().sum().sum() > 0:
            logger.warning(f"  - NaN 값 제거: {df.isnull().sum().sum()}개")
            df = df.dropna()
        return df

    X_train = pd.read_csv(DATA_SPLITS / 'train_processed.csv')
    X_train = clean_data(X_train)
    y_train = X_train['SeriousDlqin2yrs']
    X_train = X_train.drop('SeriousDlqin2yrs', axis=1)

    X_test = pd.read_csv(DATA_SPLITS / 'test_processed.csv')
    X_test = clean_data(X_test)
    y_test = X_test['SeriousDlqin2yrs']
    X_test = X_test.drop('SeriousDlqin2yrs', axis=1)

    logger.info(f"✓ 데이터 로드 완료: train={X_train.shape}, test={X_test.shape}")
    logger.info(f"  - 부도율: {y_train.mean():.2%}")

    return X_train, X_test, y_train, y_test


def calculate_ks_statistic(y_true, y_pred_proba):
    """KS Statistic 계산"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks = np.max(tpr - fpr)
    return ks


def train_no_balancing(X_train, y_train, X_test, y_test, model_name="XGBoost"):
    """1. 불균형 처리 없음"""
    logger.info("\n📊 1. No Balancing (기존 방식)")

    with mlflow.start_run(run_name=f"{model_name}_NoBalancing"):
        mlflow.log_param("method", "no_balancing")

        if model_name == "XGBoost":
            model = XGBClassifier(
                random_state=42, n_estimators=100, learning_rate=0.1,
                max_depth=6, use_label_encoder=False, eval_metric='logloss'
            )
        else:
            model = LogisticRegression(random_state=42, max_iter=1000)

        model.fit(X_train, y_train)

        # 테스트 평가
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_pred_proba)
        ks = calculate_ks_statistic(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        metrics = {'auc': auc, 'ks': ks, 'f1': f1, 'precision': precision, 'recall': recall}
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        logger.info(f"  ✓ AUC: {auc:.4f}, KS: {ks:.4f}, F1: {f1:.4f}")
        return metrics


def train_class_weight(X_train, y_train, X_test, y_test, model_name="XGBoost"):
    """2. class_weight='balanced'"""
    logger.info("\n📊 2. class_weight='balanced'")

    with mlflow.start_run(run_name=f"{model_name}_ClassWeight"):
        mlflow.log_param("method", "class_weight_balanced")

        if model_name == "XGBoost":
            # XGBoost의 경우 scale_pos_weight 사용
            neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
            model = XGBClassifier(
                random_state=42, n_estimators=100, learning_rate=0.1,
                max_depth=6, scale_pos_weight=neg_pos_ratio,
                use_label_encoder=False, eval_metric='logloss'
            )
        else:
            model = LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_pred_proba)
        ks = calculate_ks_statistic(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        metrics = {'auc': auc, 'ks': ks, 'f1': f1, 'precision': precision, 'recall': recall}
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        logger.info(f"  ✓ AUC: {auc:.4f}, KS: {ks:.4f}, F1: {f1:.4f}")
        return metrics


def train_smote(X_train, y_train, X_test, y_test, model_name="XGBoost"):
    """3. SMOTE (오버샘플링)"""
    logger.info("\n📊 3. SMOTE (오버샘플링)")

    with mlflow.start_run(run_name=f"{model_name}_SMOTE"):
        mlflow.log_param("method", "smote")

        # SMOTE 적용
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        logger.info(f"  - SMOTE 전: {(y_train == 1).sum()} 부도, {(y_train == 0).sum()} 정상")
        logger.info(f"  - SMOTE 후: {(y_train_balanced == 1).sum()} 부도, {(y_train_balanced == 0).sum()} 정상")
        mlflow.log_param("smote_samples", len(X_train_balanced))

        if model_name == "XGBoost":
            model = XGBClassifier(
                random_state=42, n_estimators=100, learning_rate=0.1,
                max_depth=6, use_label_encoder=False, eval_metric='logloss'
            )
        else:
            model = LogisticRegression(random_state=42, max_iter=1000)

        model.fit(X_train_balanced, y_train_balanced)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_pred_proba)
        ks = calculate_ks_statistic(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        metrics = {'auc': auc, 'ks': ks, 'f1': f1, 'precision': precision, 'recall': recall}
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        logger.info(f"  ✓ AUC: {auc:.4f}, KS: {ks:.4f}, F1: {f1:.4f}")
        return metrics


def train_smote_enn(X_train, y_train, X_test, y_test, model_name="XGBoost"):
    """4. SMOTE + ENN (결합)"""
    logger.info("\n📊 4. SMOTE + ENN (결합)")

    with mlflow.start_run(run_name=f"{model_name}_SMOTE_ENN"):
        mlflow.log_param("method", "smote_enn")

        # SMOTE + ENN 적용
        smote = SMOTE(random_state=42, k_neighbors=5)
        enn = EditedNearestNeighbours(n_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        X_train_balanced, y_train_balanced = enn.fit_resample(X_train_balanced, y_train_balanced)

        logger.info(f"  - 최종 샘플: {(y_train_balanced == 1).sum()} 부도, {(y_train_balanced == 0).sum()} 정상")
        mlflow.log_param("final_samples", len(X_train_balanced))

        if model_name == "XGBoost":
            model = XGBClassifier(
                random_state=42, n_estimators=100, learning_rate=0.1,
                max_depth=6, use_label_encoder=False, eval_metric='logloss'
            )
        else:
            model = LogisticRegression(random_state=42, max_iter=1000)

        model.fit(X_train_balanced, y_train_balanced)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_pred_proba)
        ks = calculate_ks_statistic(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        metrics = {'auc': auc, 'ks': ks, 'f1': f1, 'precision': precision, 'recall': recall}
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        logger.info(f"  ✓ AUC: {auc:.4f}, KS: {ks:.4f}, F1: {f1:.4f}")
        return metrics


def main():
    """메인 실행"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("="*60)
    logger.info("🚀 불균형 데이터 처리 기법 비교 실험 시작")
    logger.info("="*60)

    # MLflow 설정
    mlflow.set_experiment("imbalance_comparison")

    # 데이터 로드
    X_train, X_test, y_train, y_test = load_data()

    # 4가지 방법 비교
    all_results = {}

    for model_name in ["XGBoost", "Logistic Regression"]:
        logger.info(f"\n\n{'='*60}")
        logger.info(f"🤖 {model_name} 비교 실험")
        logger.info(f"{'='*60}")

        results = {
            'No Balancing': train_no_balancing(X_train, y_train, X_test, y_test, model_name),
            'class_weight': train_class_weight(X_train, y_train, X_test, y_test, model_name),
            'SMOTE': train_smote(X_train, y_train, X_test, y_test, model_name),
            'SMOTE+ENN': train_smote_enn(X_train, y_train, X_test, y_test, model_name),
        }

        all_results[model_name] = results

    # 결과 비교
    logger.info(f"\n\n{'='*60}")
    logger.info("📊 최종 비교 결과")
    logger.info(f"{'='*60}")

    for model_name, results in all_results.items():
        logger.info(f"\n{model_name}:")
        comparison_df = pd.DataFrame(results).T
        logger.info(comparison_df.to_string())

        # 결과 저장
        comparison_df.to_csv(RESULTS_DIR / f'{model_name}_comparison.csv')

    # 최고 성능 확인
    logger.info(f"\n\n{'='*60}")
    logger.info("🏆 최고 성능 기법")
    logger.info(f"{'='*60}")

    for model_name, results in all_results.items():
        best_method = max(results.items(), key=lambda x: x[1]['auc'])
        logger.info(f"{model_name}: {best_method[0]} (AUC: {best_method[1]['auc']:.4f})")

    logger.info(f"\n{'='*60}")
    logger.info("✅ 불균형 비교 실험 완료!")
    logger.info(f"{'='*60}")

    return all_results


if __name__ == '__main__':
    main()
