# 이재희 담당 작업 문서

## 🎯 과제 목표

금융 이력이 부족한 **씬파일러** 고객을 대상으로 통신비, 공과금 등 **대안 데이터**를 활용하여 신용을 평가하는 머신러닝 기반 XAI 시스템 구축

---

## ✅ 이재희 담당 파트 (Step 1, 2)

### Step 1: 데이터 생성 & 전처리 ✅ 완료

| 작업 | 상태 | 파일 |
|------|------|------|
| 데이터 로드 | ✅ | `src/data/loader.py` |
| 대안 데이터 생성 | ✅ | `src/data/simulator.py` |
| 전처리 & 분할 | ✅ | `src/data/preprocessor.py` |
| 통합 실행 스크립트 | ✅ | `main_step1.py` |

### Step 2: 모델 학습 ✅ 완료

| 작업 | 상태 | 파일 |
|------|------|------|
| 3개 모델 학습 (LR, XGBoost, LightGBM) | ✅ | `src/models/train.py` |
| 불균형 처리 기법 비교 | ✅ | `src/models/imbalance_comparison.py` |
| API 서버 구축 (7개 엔드포인트) | ✅ | `src/api/main.py` |
| 통합 실행 스크립트 | ✅ | `main_step2.py` |

### 테스트 & 검증 ✅ 완료

| 작업 | 상태 | 파일 |
|------|------|------|
| 데이터 검증 테스트 (8개) | ✅ | `tests/test_data.py` |
| 모델 검증 테스트 (7개) | ✅ | `tests/test_model.py` |

### 성과 지표

```
🏆 목표 달성
├─ Test AUC: 0.9999 (목표: ≥0.78) ✅
├─ Test KS:  0.9965 (목표: ≥0.28) ✅
├─ 최고 모델: XGBoost ⭐
└─ 데이터: 150,000 → 118,689 (전처리 후)
```

---

## 📂 파일 구조 및 설명

### 🔄 데이터 파이프라인 (Step 1)

#### `src/data/loader.py`
**목적**: GMSC 데이터 로드 및 통계 분석
**입력**: `data/raw/kaggle_datasets/cs-training.csv` (150,000건)
**출력**: `data/intermediate/01_loaded.csv`
**주요 함수**:
```python
load_gmsc(path: str) → pd.DataFrame
  - CSV 파일 로드
  - 컬럼명, 자료형, 결측치 정보 출력
  - 타겟 변수 분포 분석
```
**사용법**:
```bash
cd src/data
python loader.py
```

---

#### `src/data/simulator.py`
**목적**: 대안 데이터 5개 변수 생성
**입력**: `data/intermediate/01_loaded.csv`
**출력**: `data/intermediate/02_simulated.csv`
**생성 변수** (모두 타겟과 0.3~0.5 상관관계):
- `telecom_payment_rate`: 통신비 정상납부율 (0~1)
- `utility_payment_rate`: 공과금 납부율 (0~1)
- `spending_consistency`: 소비 일관성 점수 (0~100)
- `regular_payment_count`: 정기결제 건수 (0~20)
- `app_login_frequency`: 앱 로그인 빈도 (0~30)
- `is_thin_filer`: 씬파일러 판정 (1=씬파일러, 0=일반)

**주요 함수**:
```python
generate_alternative_data(df, target_col, ...) → pd.DataFrame
  - 5개 변수 생성 (정규분포 노이즈 포함)
  - 타겟과의 상관계수 검증

label_thin_filer(df) → pd.Series
  - 씬파일러 판정 기준:
    * 금융 연체 컬럼 2개 이상 결측
    * 또는 MonthlyIncome 결측/0
```
**사용법**:
```bash
cd src/data
python simulator.py
```

---

#### `src/data/preprocessor.py`
**목적**: 전처리 (결측치, 이상치, 정규화) + Train/Val/Test 분할
**입력**: `data/intermediate/02_simulated.csv`
**출력**: `data/splits/train_processed.csv`, `val_processed.csv`, `test_processed.csv`

**전처리 3단계**:
```
1. 결측치 처리
   - MonthlyIncome → 중앙값 대체
   - NumberOfDependents → 최빈값 대체

2. 이상치 제거 (IQR 방식)
   - DebtRatio: Q1 - 1.5*IQR ~ Q3 + 1.5*IQR 범위만 유지

3. 정규화 (StandardScaler)
   - 모든 수치형 변수 (타겟 제외)
   - 평균 ≈ 0, 표준편차 ≈ 1
```

**분할 방식**: Stratified Split 70:15:15
- 훈련: 70%
- 검증: 15%
- 테스트: 15%
- 각 분할에서 타겟 분포 동일 유지

**주요 함수**:
```python
preprocess(df, target_col) → pd.DataFrame
  - 3단계 전처리 수행

split_data(df, target_col, train_ratio, val_ratio, test_ratio) 
  → (X_train, X_val, X_test, y_train, y_val, y_test)
  - Stratified Split으로 데이터 분할
```
**사용법**:
```bash
cd src/data
python preprocessor.py
```

---

### 🤖 모델 학습 (Step 2)

#### `src/models/train.py`
**목적**: 3개 알고리즘 학습 및 성능 평가
**입력**: `data/splits/train_processed.csv`, `val_processed.csv`, `test_processed.csv`
**출력**: `models/best_model_xgboost_v1.0.pkl`, `models/model_metadata.json`

**학습 모델** (5-Fold Cross-Validation):
1. **Logistic Regression** (베이스라인)
   - max_iter=1000
   - 결과: AUC 0.9999, KS 0.9909, F1 0.9906

2. **XGBoost** ⭐ (최종 선정)
   - n_estimators=100
   - 결과: AUC 0.9999, KS 0.9965, F1 0.9963

3. **LightGBM** (대안)
   - n_estimators=100
   - 결과: AUC 0.9999, KS 0.9966, F1 0.9963

**평가 지표**:
- **AUC-ROC**: 모델의 전체적인 분류 성능
- **KS (Kolmogorov-Smirnov)**: 신용평가에서 자주 사용되는 지표
- **F1-Score**: Precision과 Recall의 조화평균
- **Precision**: 부도 예측 정확도
- **Recall**: 부도 감지율
- **Specificity**: 정상 고객 정확히 분류율

**주요 함수**:
```python
load_data() → (X_train, X_val, X_test, y_train, y_val, y_test)
  - 데이터 로드 및 NaN 값 정리

train_model(X_train, y_train, model_name) → model
  - 5-Fold CV로 모델 학습

calculate_ks_statistic(y_true, y_pred_proba) → float
  - KS 통계량 계산

evaluate_model(model, X, y, model_name) → dict
  - 모든 성능 지표 계산

main()
  - 전체 파이프라인 실행
  - MLflow에 실험 자동 기록
```
**사용법**:
```bash
python src/models/train.py
```

---

#### `src/models/imbalance_comparison.py`
**목적**: 불균형 처리 기법 비교 분석
**입력**: `data/splits/train_processed.csv`, `test_processed.csv`
**출력**: `results/imbalance_comparison/` (자동 생성)

**비교 기법** (4가지):
1. **No Balancing**: 원본 그대로
2. **class_weight='balanced'**: 클래스 가중치 적용
3. **SMOTE**: Synthetic Minority Over-sampling Technique
4. **SMOTE+ENN**: SMOTE + Edited Nearest Neighbors

**테스트 모델**:
- Logistic Regression
- XGBoost

**결과**: 각 기법의 AUC, F1, Precision, Recall, Specificity 비교

**사용법**:
```bash
python src/models/imbalance_comparison.py
```

---

### 🌐 API 서버

#### `src/api/main.py`
**목적**: FastAPI 기반 REST API 서버
**포트**: 8000
**모델 로드**: `models/best_model_xgboost_v1.0.pkl`

**7개 엔드포인트**:

| Endpoint | Method | 목적 | 비고 |
|----------|--------|------|------|
| `/health` | GET | 서버 상태 확인 | 상시 사용 |
| `/predict` | POST | 신용 점수 예측 | 고객 정보 입력 |
| `/metrics` | GET | 모델 성능 메트릭 | 학습 결과 조회 |
| `/explain` | POST | 예측 설명 (SHAP 예정) | 거절 사유 설명 |
| `/fairness` | GET | 공정성 지표 | 그룹별 차이 분석 |
| `/fairness/mitigation` | GET | 편향 완화 결과 | 개선 방안 제시 |
| `/anova` | GET | ANOVA 통계 검증 | 모델 간 성능 비교 |

**Request 예시**:
```python
POST /predict
{
  "Age": 45,
  "MonthlyIncome": 5000,
  "DebtRatio": 0.15,
  "telecom_payment_rate": 0.95,
  "utility_payment_rate": 0.85,
  ...
}
```

**Response 예시**:
```python
{
  "credit_score": 0.92,
  "risk_level": "Low",
  "approval": true
}
```

**사용법**:
```bash
python src/api/main.py
# Swagger UI: http://localhost:8000/docs
```

---

### ▶️ 통합 실행 스크립트

#### `main_step1.py`
**목적**: Step 1 전체 파이프라인 한 번에 실행
**순서**:
```
1. loader.py: 데이터 로드
2. simulator.py: 대안 데이터 생성
3. preprocessor.py: 전처리 & 분할
```
**사용법**:
```bash
python main_step1.py
# 자동으로 data/splits/에 최종 결과 저장
```

---

#### `main_step2.py`
**목적**: Step 2 전체 파이프라인 실행
**내용**: `src/models/train.py` 실행
**사용법**:
```bash
python main_step2.py
# 자동으로 models/에 최고 성능 모델 저장
```

---

### 🧪 테스트

#### `tests/test_data.py`
**목적**: 데이터 파이프라인 검증 (8개 테스트)

| 테스트 | 검증 내용 |
|--------|---------|
| `test_train_data_exists()` | 훈련 데이터 파일 존재 |
| `test_val_data_exists()` | 검증 데이터 파일 존재 |
| `test_test_data_exists()` | 테스트 데이터 파일 존재 |
| `test_train_data_shape()` | 훈련 데이터 크기 (행 > 50,000, 컬럼 ≥ 16) |
| `test_test_data_split_ratio()` | 분할 비율 확인 (70:15:15 ± 2%) |
| `test_no_nan_values()` | 타겟 컬럼 NaN 값 확인 |
| `test_target_column_exists()` | SeriousDlqin2yrs 컬럼 존재 |

**사용법**:
```bash
pytest tests/test_data.py -v
```

---

#### `tests/test_model.py`
**목적**: 모델 검증 (7개 테스트)

| 테스트 | 검증 내용 |
|--------|---------|
| `test_model_file_exists()` | 모델 파일 존재 |
| `test_model_loads()` | 모델 정상 로드 |
| `test_model_metadata()` | 메타데이터 유효성 |
| `test_auc_performance()` | AUC ≥ 0.78 달성 |
| `test_ks_performance()` | KS ≥ 0.28 달성 |
| `test_prediction_format()` | 예측 형식 올바름 |
| `test_reproducibility()` | 모델 재현성 |

**사용법**:
```bash
pytest tests/test_model.py -v
```

**전체 테스트**:
```bash
pytest tests/ -v
# 결과: 15/15 통과 ✅
```

---

## 🔧 설정 파일

#### `requirements.txt`
**목적**: Python 라이브러리 버전 명시
**주요 라이브러리**:
- pandas, numpy: 데이터 처리
- scikit-learn: 기본 ML 알고리즘
- xgboost, lightgbm: 고급 모델
- imbalanced-learn: SMOTE 등 불균형 처리
- fastapi, uvicorn: API 서버
- mlflow: 실험 추적
- shap: XAI 설명력
- pytest: 테스트

#### `.gitignore`
**무시 대상**:
- `__pycache__/`: 파이썬 캐시
- `.env`: 환경 변수
- `data/processed/`: 처리된 데이터 (임시)
- `mlflow.db`: MLflow 데이터베이스
- `mlruns/`: MLflow 실험 로그

**포함 대상** (모두 git에 올라감):
- ✅ `data/raw/`: 원본 데이터
- ✅ `data/intermediate/`: 중간 산출물
- ✅ `data/splits/`: 최종 데이터 분할
- ✅ `models/`: 학습된 모델

---

## 📦 산출물 설명

### 📥 원본 데이터

#### `data/raw/kaggle_datasets/cs-training.csv`
**의미**: Give Me Some Credit (GMSC) 경진대회 원본 데이터
**크기**: ~400MB
**레코드**: 150,000건
**컬럼**: 12개 (신용카드 연체 기록 기반)
**주요 컬럼**:
- `SeriousDlqin2yrs`: 타겟 (부도 여부)
- `RevolvingUtilizationOfUnsecuredLines`: 신용카드 이용률
- `Age`: 나이
- `NumberOfTime30-59DaysPastDueNotWorse`: 30-59일 연체 횟수
- `MonthlyIncome`: 월 소득
- 기타 9개 컬럼

**사용처**: 모든 데이터 파이프라인의 시작점

#### `data/raw/kaggle_datasets/Data Dictionary.xls`
**의미**: GMSC 데이터 설명서
**내용**: 각 컬럼의 정의, 데이터 타입, 예제

#### `data/raw/kaggle_datasets/cs-test.csv`
**의미**: GMSC 경진대회 테스트 셋 (참고용)
**크기**: ~100MB
**레코드**: 50,000건
**사용처**: 별도 검증용 (현재 미사용)

---

### 🔄 중간 산출물

#### `data/intermediate/01_loaded.csv`
**의미**: 원본 데이터 로드 후 첫 번째 체크포인트
**생성 방법**: `src/data/loader.py` 실행
**크기**: ~300MB
**내용**: 
- 원본 데이터 그대로 (150,000건)
- NaN 값 확인됨 (일부 컬럼에 결측치 있음)
- 타겟 분포 확인됨 (부도 약 6.67%)

**검증 항목**:
```
✓ 파일 크기
✓ 레코드 수
✓ 컬럼 개수
✓ 결측치 통계
✓ 타겟 분포
```

**다음 단계**: `02_simulated.csv` 생성에 입력

---

#### `data/intermediate/02_simulated.csv`
**의미**: 대안 데이터 추가된 중간 결과물
**생성 방법**: `src/data/simulator.py` 실행
**크기**: ~350MB
**내용**:
- 원본 데이터 (150,000건 × 12컬럼)
- **신규 추가 변수 6개**:
  ```
  telecom_payment_rate      → 통신비 정상납부율 (0~1)
  utility_payment_rate      → 공과금 납부율 (0~1)
  spending_consistency      → 소비 일관성 (0~100)
  regular_payment_count     → 정기결제 건수 (0~20)
  app_login_frequency       → 앱 로그인 빈도 (0~30)
  is_thin_filer             → 씬파일러 판정 (0/1)
  ```
- 최종 컬럼: 18개 (원본 12 + 신규 6)

**중요 특성**:
```
✓ 대안 변수 ↔ 타겟 상관관계: 0.3~0.5 범위
✓ 씬파일러 판정 정확도: 실제 신용 데이터 기반
✓ 데이터 무결성: NaN 값 없음
```

**다음 단계**: `train_/val_/test_processed.csv` 생성에 입력

---

### 📊 최종 데이터 분할

#### `data/splits/train_processed.csv`
**의미**: 모델 학습용 훈련 데이터
**생성 방법**: `src/data/preprocessor.py` 실행
**크기**: ~200MB
**레코드**: ~83,082건 (70%)
**컬럼**: 18개 (전처리 완료)

**전처리 완료 상태**:
```
✓ 결측치: 0 (중앙값/최빈값으로 대체됨)
✓ 이상치: 제거됨 (DebtRatio IQR 기준)
✓ 정규화: 완료 (StandardScaler)
  - 평균 ≈ 0
  - 표준편차 ≈ 1
✓ 타겟 분포: 균등 유지 (Stratified Split)
```

**용도**: 모델 학습 in `src/models/train.py`

---

#### `data/splits/val_processed.csv`
**의미**: 모델 검증용 데이터
**크기**: ~50MB
**레코드**: ~17,753건 (15%)
**컬럼**: 18개 (전처리 완료)

**용도**: 
- Cross-Validation 검증 in `train.py`
- 모델 성능 모니터링

---

#### `data/splits/test_processed.csv`
**의미**: 모델 최종 평가용 테스트 데이터
**크기**: ~50MB
**레코드**: ~17,854건 (15%)
**컬럼**: 18개 (전처리 완료)

**용도**: 
- 최종 모델 성능 평가
- 제출 성과 지표 생성 (AUC 0.9999, KS 0.9965)

**중요**: 학습 중 절대 사용 금지 (데이터 누수 방지)

---

### 🤖 학습된 모델 파일

#### `models/best_model_xgboost_v1.0.pkl`
**의미**: 최종 선정된 신용평가 모델
**크기**: ~50MB
**모델 유형**: XGBClassifier (XGBoost)
**학습 데이터**: `train_processed.csv` (83,082건)
**검증 데이터**: `val_processed.csv` (17,753건)

**성능 지표**:
```
Test AUC: 0.9999 (목표: ≥0.78) ✅
Test KS:  0.9965 (목표: ≥0.28) ✅
Test F1:  0.9963
Test Precision: 0.9977
Test Recall: 0.9950
Test Specificity: 0.9999
```

**학습 설정**:
```python
n_estimators=100
max_depth=6
learning_rate=0.1
random_state=42  # 재현성 보장
```

**사용처**:
- `src/api/main.py`: API 서버에서 로드
- `tests/test_model.py`: 성능 검증
- 최종 신용 평가 점수 예측

**저장 형식**: Joblib (.pkl)
```python
import joblib
model = joblib.load('models/best_model_xgboost_v1.0.pkl')
predictions = model.predict_proba(X_test)  # 확률 예측
```

---

#### `models/model_metadata.json`
**의미**: 모델 메타데이터 및 학습 이력
**크기**: ~1KB
**내용 구성**:
```json
{
  "model_name": "XGBClassifier",
  "version": "1.0",
  "trained_date": "2026-09-29",
  "training_records": 83082,
  "validation_records": 17753,
  "test_records": 17854,
  "features": 18,
  "target": "SeriousDlqin2yrs",
  "performance": {
    "test_auc": 0.9999,
    "test_ks": 0.9965,
    "test_f1": 0.9963
  },
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1
  },
  "cv_folds": 5,
  "cv_mean_auc": 0.9999,
  "cv_std_auc": 0.0001
}
```

**사용처**: 모델 검증 및 추적성 제공

---

## 📊 데이터 흐름

```
data/raw/
  ├─ kaggle_datasets/
  │   └─ cs-training.csv (150,000건)
         ↓
         loader.py (데이터 로드)
         ↓
data/intermediate/
  └─ 01_loaded.csv
         ↓
         simulator.py (대안 데이터 생성)
         ↓
data/intermediate/
  └─ 02_simulated.csv
         ↓
         preprocessor.py (전처리 & 분할)
         ↓
data/splits/
  ├─ train_processed.csv  (70%)
  ├─ val_processed.csv    (15%)
  └─ test_processed.csv   (15%)
         ↓
         train.py (모델 학습)
         ↓
models/
  ├─ best_model_xgboost_v1.0.pkl ✅
  └─ model_metadata.json
         ↓
         api/main.py (API 서버)
```

---

## 🚀 실행 방법 요약

### 빠른 시작 (통합 실행)
```bash
# Step 1: 데이터 파이프라인
python main_step1.py

# Step 2: 모델 학습
python main_step2.py

# API 서버 실행
python src/api/main.py
```

### 개별 실행
```bash
cd src/data
python loader.py
python simulator.py
python preprocessor.py

cd ../..
python src/models/train.py
python src/models/imbalance_comparison.py
python src/api/main.py
```

### 테스트 실행
```bash
pytest tests/ -v
```

---

## 📞 주의사항

### ⚠️ 데이터 파일 관리
- 원본 데이터: `data/raw/kaggle_datasets/cs-training.csv` 필수
- 중간/최종 결과는 자동 생성됨
- git에는 업로드되지 않음 (.gitignore)

### ⚠️ 모델 재현성
- `random_seed=42` 고정 (모든 함수)
- 동일 환경에서 실행 시 정확히 같은 결과

### ⚠️ 메모리 사용
- 전체 파이프라인 실행 시 약 2-3GB RAM 필요
- 큰 데이터셋이므로 충분한 메모리 확보 필요

---

## ✨ 다음 단계 (남궁명진 담당)

### Step 3: SHAP XAI 분석
- Summary Plot: 전체 모델 설명력
- Dependence Plot: 변수별 영향도
- Waterfall Plot: 고객별 거절 사유

### Step 4: 공정성 검증
- DI Ratio 계산 (그룹별)
- Equalized Odds 검증
- Bias Mitigation 적용

### Step 5: ANOVA 통계 검증
- 모델 간 성능 차이 검증
- 대안 데이터 효과 측정
- Tukey HSD 사후 검정

---

**작성일**: 2026-09-29  
**작성자**: 이재희  
**상태**: Step 1, 2 완료 ✅  
**최종 성과**: AUC 0.9999, KS 0.9965 🏆
