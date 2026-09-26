# XAI 기반 대안신용평가 시스템

신용불량단 팀 - 코디세이 WebApp PBL 학기 프로젝트

## 프로젝트 개요

금융 이력이 부족한 씬파일러(금융 연체 관련 컬럼 2개 이상 결측 또는 신용카드 거래 이력 12개월 미만)를 대상으로, 통신비/공과금 납부율 등 비금융 대안 데이터를 활용해 신용을 평가하는 시스템.
GMSC(Give Me Some Credit) 데이터셋 기반 모델 학습, SHAP 기반 XAI 설명, 공정성(Fairness) 검증, ANOVA 통계 검증을 거쳐 FastAPI + Streamlit 구조로 서비스한다.

## 팀 구성원 역할

- **이재희** - 데이터·모델 (데이터 생성, 전처리, 모델 학습 3종)
- **이하늘** - 백엔드·인프라 (API/DB 설계, API 개발, Docker Compose)
- **이윤아** - 프론트엔드 (화면 설계, Streamlit 대시보드, 시각화)
- **남궁명진** - 모델 분석 (성능 비교, XAI·SHAP, 공정성·통계 검증)

## 디렉토리 구조

```
xai-credit-scoring/
├── data/
│   ├── raw/            # 원본 데이터
│   │   └── kaggle_datasets/  # Give Me Some Credit 캐글 데이터셋 (git 포함)
│   ├── processed/      # 전처리 완료 데이터 (git 미포함)
│   ├── splits/         # Train/Val/Test 분할 결과 (git 미포함)
│   └── mock/           # 프론트 개발용 mock 데이터 (git 포함)
├── src/
│   ├── data/
│   │   ├── loader.py         # GMSC/German Credit 로드
│   │   ├── simulator.py      # 대안 데이터(통신비/공과금 등) 시뮬레이터
│   │   └── preprocessor.py   # 결측치 처리, 스케일링, Stratified Split
│   ├── models/          # 모델 학습 코드 (예정)
│   ├── explainer/       # SHAP 기반 XAI 설명 코드 (예정)
│   ├── fairness/        # 공정성 지표·편향 완화 코드 (예정)
│   ├── statistics/
│   │   └── anova.py     # One-way ANOVA + Tukey HSD 사후 검정
│   ├── api/
│   │   └── main.py      # FastAPI 서버 (7개 엔드포인트, 현재 mock 응답)
│   └── dashboard/
│       └── app.py       # Streamlit 대시보드 (5개 탭, API와 HTTP 통신만)
├── models/              # 학습된 모델 아티팩트 (git 미포함)
├── notebooks/           # 탐색적 분석용 노트북
├── scripts/
│   └── generate_mock_data.py   # mock 데이터 생성 스크립트
├── tests/               # 테스트 코드
├── docs/                # 프로젝트 문서
├── docker-compose.yml   # api(8000) / dashboard(8501) / mlflow(5000) 3개 서비스
├── Dockerfile
├── requirements.txt
└── .env.example
```

## 현재 진행 상황

- API 서버(`src/api/main.py`)는 7개 엔드포인트(`/health`, `/predict`, `/explain`, `/fairness`, `/fairness/mitigation`, `/metrics`, `/anova`) 모두 구현 완료. 단, 실제 모델 학습 전이라 규칙 기반 mock 응답을 반환한다. 입력값에 따라 점수·등급·SHAP 기여도가 달라지긴 하지만 실제 예측값은 아님.
- Streamlit 대시보드(`src/dashboard/app.py`)는 미션 스펙의 5개 탭(고객심사/XAI분석/공정성/모델성능/통계검증) 구조를 갖추고, API 서버하고만 통신하도록 구성됨 (모델 파일 직접 로드 없음 - MSA 제약 준수).
- `src/data/loader.py`, `src/data/simulator.py`, `src/data/preprocessor.py`는 함수 시그니처만 정의된 상태 - 이재희가 실제 로직 구현 예정.
- `src/models/`는 3가지 모델 학습 (Logistic Regression, XGBoost, LightGBM) - 이재희가 구현 예정.
- `src/statistics/anova.py`, `src/fairness/`, `src/explainer/`는 함수 시그니처만 정의된 상태 - 남궁명진이 실제 로직 구현 예정.
- GMSC 데이터셋(`data/raw/kaggle_datasets/`)이 추가됨. 포함 파일: `cs-training.csv`(훈련 데이터), `cs-test.csv`(테스트 데이터), `sampleEntry.csv`(샘플 제출), `Data Dictionary.xls`(데이터 사전).
- 모델 학습이 끝나면 `src/api/main.py`의 mock 로직만 실제 모델/SHAP/Fairlearn 호출로 교체하면 되고, 엔드포인트·응답 스키마는 그대로 유지하면 됨.

## 실행 방법

### 로컬 실행

```bash
pip install -r requirements.txt

# 터미널 1 - API 서버
uvicorn src.api.main:app --reload
# http://localhost:8000/docs 에서 Swagger UI 확인 가능

# 터미널 2 - 대시보드
streamlit run src/dashboard/app.py
# http://localhost:8501
```

### Docker Compose

```bash
docker-compose up
```

- api: http://localhost:8000
- dashboard: http://localhost:8501
- mlflow: http://localhost:5000

## 환경 변수 설명

`.env.example` 참고. 실제 실행 시 `.env`로 복사해서 사용.

| 변수 | 설명 |
|---|---|
| DATA_DIR | 데이터 디렉토리 경로 |
| RANDOM_SEED | 재현성을 위한 랜덤 시드 |
| THIN_FILER_RATIO | 시뮬레이터의 씬파일러 비율 조정 파라미터 |
| BIAS_RATIO | 시뮬레이터의 성별/연령대별 승인율 편향 조정 파라미터 |
| API_PORT | FastAPI 서버 포트 |
| DASHBOARD_PORT | Streamlit 대시보드 포트 |
| MLFLOW_PORT | MLflow UI 포트 |
