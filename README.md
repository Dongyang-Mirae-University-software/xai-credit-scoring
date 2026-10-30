# XAI 기반 대안신용평가 시스템

신용불량단 팀 - 코디세이 WebApp PBL 학기 프로젝트

> **본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.**
>
> 같은 문구가 대시보드 모든 페이지(사이드바 하단), API 응답의 `disclaimer` 필드, 실험 리포트에 들어 있습니다.

## 프로젝트 개요

금융 이력이 부족해 기존 CB(신용평가) 모델로는 평가가 어려운 **씬파일러**를 위해, 통신비·공과금 납부율 같은 비금융 **대안 데이터**를 더해 신용을 평가하는 시스템입니다.
Give Me Some Credit(GMSC) 데이터에 대안 데이터 시뮬레이터를 결합해 모델을 학습하고, SHAP 설명(XAI)·공정성 검증·ANOVA 통계 검증을 거쳐 FastAPI(API)와 Streamlit(대시보드)으로 제공합니다. 모든 실험은 MLflow로 기록합니다.

### 데이터
| 항목 | 값 |
|---|---|
| 원본 | GMSC `data/raw/cs-training.csv` 150,000명 |
| 부도율 (2년 내 90일 이상 연체) | 6.68% (Positive:Negative ≈ 1:14) |
| 대안 변수 5개 | 통신비 정상납부율, 공과금 납부율, 소비 일관성, 정기결제 건수, 앱 로그인 빈도 (`src/data/simulator.py`) |
| 씬파일러 | 4,110명 (2.74%) |
| 분할 | Train / Validation / Test = 70 : 15 : 15 (stratify, 시드 42) |

**씬파일러 판정 기준** (`label_thin_filer`): 과제 기준인 "연체 컬럼 2개 이상 결측"은 GMSC에 해당자가 없어서, 아래 기준을 함께 씁니다 (기준 1 OR 기준 2).
- 기준 1: 연체 컬럼 2개 이상 결측
- 기준 2: 보유 대출·신용 한도 1개 이하 **이고** 부동산 담보대출 0개 **이고** 연체 기록 0회

### 모델과 검증
- 모델: Logistic Regression · XGBoost · LightGBM × 불균형 처리(class_weight / SMOTE) × 데이터(전통 / 대안 / 통합), 5-Fold Stratified CV
- 최종 모델 선택: Validation AUC 최고 모델 → Test에서 1회만 평가 (현재 결과는 [`models/README.md`](models/README.md))
- 검증: 성능(AUC·KS·PSI·씬파일러 AUC 향상), SHAP 전역·지역 해석과 거절 사유, 공정성(DI Ratio·Equalized Odds, 편향 완화 전/후), ANOVA 2종 + Tukey HSD

## 팀 구성원 역할

| 이름 | 역할 | 담당 |
|---|---|---|
| 이재희 | 데이터·모델 | 데이터 로드, 대안 데이터 시뮬레이터, 전처리·분할, 모델 학습 (`main_step1·2`) |
| 이하늘 | 백엔드·인프라 | FastAPI·Pydantic 스키마, Docker Compose, MLflow 서버 |
| 이윤아 | 프론트엔드 | 화면 설계, Streamlit 대시보드 5개 페이지, 시각화 |
| 남궁명진 | 모델 분석 | 성능 비교, XAI·SHAP, 공정성·통계 검증 (`main_step3~6`), 실험 리포트 |

## 실행 방법

### 1. Docker Compose (권장)

```bash
git clone https://github.com/Dongyang-Mirae-University-software/xai-credit-scoring.git
cd xai-credit-scoring
docker compose up --build     # .env 없이도 뜬다. 처음 한 번은 이미지 빌드에 몇 분 걸림
```

| 서비스 | 주소 | 설명 |
|---|---|---|
| API | http://localhost:8000/docs | Swagger UI. 화면별 태그로 엔드포인트 확인 |
| 대시보드 | http://localhost:8501 | 고객 심사 / XAI 분석 / 공정성 / 모델 성능 / 통계 검증 |
| MLflow UI | http://localhost:5000 | 이 PC의 `./mlruns` 기록 (팀 서버 기록이 아님) |

- 최종 모델·스케일러는 `models/`에 들어 있어서 clone 직후 바로 `/predict`·`/explain`이 동작합니다.
- `/metrics`·`/fairness`·`/anova`·`/explain/summary`는 아래 파이프라인이 만든 `outputs/` 결과를 돌려줍니다. `outputs/`가 없으면 내장 예시(`src/api/examples/`)를 돌려주며, `GET /health`의 `reports`에서 어느 쪽인지 확인할 수 있습니다.

### 2. 로컬 실행 (Python 3.10 권장, Docker 이미지와 같은 버전)

```bash
pip install -r requirements.txt

# 터미널 1 - API
uvicorn src.api.main:app --port 8000

# 터미널 2 - 대시보드
streamlit run src/dashboard/app.py
```

API 없이 대시보드 화면만 볼 때는 `USE_MOCK=true streamlit run src/dashboard/app.py`.

### 3. 파이프라인 (데이터 → 모델 → 검증)

```bash
python main_step1.py   # 데이터 로드 · 대안 데이터 생성 · 전처리 · 분할 → data/splits/
python main_step2.py   # 알고리즘 3 × 불균형 처리 2 × 데이터 3 = 18개 학습, Val AUC로 최종 모델 선택 → models/
python main_step3.py --model models/lightgbm_is_unbalance_integrated_v1.0.joblib \
    --baseline-model models/lightgbm_is_unbalance_financial_only_v1.0.joblib   # 성능 · SHAP → outputs/evaluation/
python main_step4.py --model models/lightgbm_is_unbalance_integrated_v1.0.joblib --reweighting   # 공정성 → outputs/fairness/
python main_step5.py   # ANOVA (MLflow의 폴드별 AUC 사용) → outputs/statistics/
python main_step6.py   # 검증 결과 요약(쉬운 영어) → outputs/mlflow_summary/  (--upload 를 주면 MLflow 서버에 올림)
```

- `--baseline-model`(전통 데이터만 쓴 모델)은 저장소에 없고 `main_step2.py`를 실행하면 생깁니다.
- 실험 기록 위치는 `MLFLOW_TRACKING_URI`로 정합니다. 비워 두면 로컬 `./mlruns`에 기록하고, Docker Compose의 MLflow UI(5000)에서 보려면 `MLFLOW_TRACKING_URI=http://localhost:5000`으로 실행합니다.
- `main_step2.py`는 `.env`를 읽습니다. `.env`에 팀 서버 주소가 있으면 팀 서버에 기록되니, 개인 실험은 위처럼 로컬 주소를 지정합니다.

### 4. 테스트

```bash
pytest
```

## 디렉토리 구조

```
xai-credit-scoring/
├── main_step1.py ~ main_step6.py   # 파이프라인 단계별 실행 스크립트 (위 "파이프라인" 참고)
├── data/
│   ├── raw/                 # GMSC 원본 (cs-training.csv, cs-test.csv, Data Dictionary.xls)
│   ├── intermediate/        # step1 중간 결과 (01_loaded.csv, 02_simulated.csv) - git 미포함
│   ├── splits/              # Train/Validation/Test 분할 CSV - git 미포함
│   └── mock/                # 개발용 mock 데이터
├── src/
│   ├── data/                # loader.py(로드) · simulator.py(대안 데이터·씬파일러) · preprocessor.py(결측·스케일링·분할)
│   ├── models/              # train.py(9개 조합 학습·5-Fold CV·최종 선택) · imbalance_comparison.py
│   ├── explainer/           # performance.py(AUC·KS·PSI·씬파일러 향상) · shap_explainer.py · reasons.py(거절 사유 문장)
│   ├── fairness/            # metrics.py(DI Ratio·Equalized Odds) · mitigation.py(편향 완화) · library_check.py(Fairlearn 교차 확인)
│   ├── statistics/          # anova.py(One-way ANOVA 2종 + Tukey HSD)
│   ├── tracking/            # MLflow 기록 도우미 (run 설명·REST)
│   ├── api/                 # FastAPI: main.py · routers/ · schemas.py · model.py · reports.py · examples/
│   └── dashboard/           # Streamlit: app.py · views/(5개 페이지) · components/ · api_client.py · config.py · mock/
├── models/                  # 최종 모델 · 스케일러 · model_metadata.json (변경 이력은 models/README.md)
├── outputs/                 # step3~6 결과 JSON·그림 (실행하면 생김, 저장소에는 없음)
├── notebooks/01_eda.ipynb   # 탐색적 데이터 분석
├── docs/                    # report_draft.md(실험 리포트 초안) · column_spec.md(칸 이름·단위)
├── tests/                   # pytest 단위 테스트
├── scripts/                 # mock 데이터 생성 등 보조 스크립트
├── mlflow/                  # 팀 MLflow 서버 배포 설정 (서버용, 로컬 실행에는 필요 없음)
├── docker-compose.yml       # api(8000) · dashboard(8501) · mlflow(5000)
├── Dockerfile
├── requirements.txt
└── .env.example
```

## 환경 변수 설명

`.env.example`을 `.env`로 복사해서 씁니다. **`.env`는 git에 올리지 않고, 비밀번호는 `.env`에만 적습니다.** 모든 값에 기본값이 있어 `.env` 없이도 실행됩니다.

### `.env.example`에 있는 값
| 변수 | 기본값 | 설명 |
|---|---|---|
| `API_PORT` | 8000 | Docker Compose에서 API를 열 PC 포트 |
| `DASHBOARD_PORT` | 8501 | Docker Compose에서 대시보드를 열 PC 포트 |
| `MLFLOW_PORT` | 5000 | Docker Compose에서 MLflow UI를 열 PC 포트 |
| `MLFLOW_TRACKING_URI` | (없음 → 로컬 `./mlruns`) | 실험을 기록·조회할 MLflow 서버 주소 |
| `MLFLOW_TRACKING_USERNAME` | (없음) | MLflow 서버 로그인 아이디 |
| `MLFLOW_TRACKING_PASSWORD` | (없음) | MLflow 서버 비밀번호. `.env`에만 적기 |
| `MLFLOW_EXPERIMENT_NAME` | `xai-credit-scoring` | 실험 이름 (`main_step5.py`가 이 실험에서 폴드별 AUC를 읽음) |

시뮬레이터 설정(씬파일러 비율·편향·시드)은 환경 변수가 아니라 `main_step1.py`에 고정되어 있습니다: `thin_filer_ratio=None`, `bias_ratio=0`, `random_seed=42`.

### 대시보드 (`src/dashboard/config.py`)
| 변수 | 기본값 | 설명 |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | 대시보드가 부를 API 주소 (Docker Compose에서는 자동으로 `http://api:8000`) |
| `USE_MOCK` | `false` | `true`면 API 대신 `mock/*.json`을 씀 (화면만 볼 때) |
| `API_TIMEOUT` / `HEALTH_TIMEOUT` | 5 / 2 | API 응답 대기 시간(초) |
| `HEALTH_CACHE_TTL` | 10 | 서버 상태 카드 새로고침 간격(초) |
| `XAI_RISK_HIGH` / `XAI_RISK_MID` | 0.7 / 0.4 | 거절 사유 카드의 고·중위험 표시 기준 (SHAP 값) |
| `WATERFALL_MAX_DISPLAY` | 8 | Waterfall에 따로 보여 줄 변수 수 |
| `ADDITIVITY_TOLERANCE` | 0.005 | SHAP 가법성 검증 허용 오차 |

### API (`src/api/`)
| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_PATH` | `models/lightgbm_is_unbalance_integrated_v1.0.joblib` | 최종 모델 파일 |
| `SCALER_PATH` | `models/preprocess_scaler_v1.0.joblib` | 전처리 스케일러 파일 |
| `MODEL_STAGE` | `Staging` | `/health`에 표시할 모델 단계 |
| `OUTPUTS_DIR` | `outputs` | 검증 결과 JSON을 읽을 폴더 |

### 공통 기준값 (API·대시보드·검증)
| 변수 | 기본값 | 설명 |
|---|---|---|
| `APPROVAL_THRESHOLD` | 0.5 | 부도 확률이 이 값 미만이면 승인 |
| `TARGET_AUC` / `TARGET_KS` / `TARGET_PSI` / `TARGET_THIN_UPLIFT` | 0.78 / 0.28 / 0.1 / 0.03 | 성능 목표 (과제 기준) |
| `FAIRNESS_DI_MIN` / `FAIRNESS_DI_MAX` / `FAIRNESS_EO_MAX` | 0.8 / 1.25 / 0.10 | 공정성 기준 (DI Ratio 범위, Equalized Odds 차이) |
| `FAIRNESS_PROTECTED_GENDER` / `FAIRNESS_PROTECTED_AGE` | `female` / `55+` | DI Ratio 계산 시 보호 집단 |
| `ANOVA_ALPHA` / `ANOVA_N_FOLDS` | 0.05 / 5 | ANOVA 유의수준, 폴드 수 |

### 검증 스크립트 (`main_step3~6`, `src/explainer`)
| 변수 | 기본값 | 설명 |
|---|---|---|
| `SPLITS_DIR` | `data/splits` | 분할 CSV 폴더 |
| `RAW_DATA` | `data/intermediate/02_simulated.csv` | 거절 사유 문장에 쓸 원래 값 |
| `BASELINE_MODEL` | (없음) | 씬파일러 AUC 향상 비교용 모델 (`--baseline-model`과 같음) |
| `SHAP_SAMPLE` | 2000 | SHAP 계산에 쓸 Test 표본 수 |
| `UPLIFT_BOOTSTRAP` | 1000 | 씬파일러 AUC 향상 신뢰구간 부트스트랩 횟수 |
| `REFERENCE_TRIM` | 0.01 | "동일 연령대 평균" 계산 시 양끝 제외 비율 |
| `ANOVA_TEST2_DATA_TYPE` | `integrated` | ANOVA 검증 2(알고리즘별)에 쓸 데이터 |
| `EVAL_OUTPUT_DIR` / `FAIRNESS_OUTPUT_DIR` / `ANOVA_OUTPUT_DIR` / `MLFLOW_SUMMARY_DIR` | `outputs/…` | 결과 저장 폴더 |
| `PLOT_FONT` | (자동 탐색) | 그림의 한글 글꼴 경로 |
