# XAI 기반 대안신용평가 시스템

신용불량단 팀 - 코디세이 WebApp PBL 학기 프로젝트

## 프로젝트 개요

금융 이력이 부족한 씬파일러(금융 연체 관련 컬럼 2개 이상 결측 또는 신용카드 거래 이력 12개월 미만)를 대상으로, 통신비/공과금 납부율 등 비금융 대안 데이터를 활용해 신용을 평가하는 시스템.
GMSC(Give Me Some Credit) 데이터셋 기반 모델 학습, SHAP 기반 XAI 설명, 공정성(Fairness) 검증, ANOVA 통계 검증을 거쳐 FastAPI + Streamlit 구조로 서비스한다.

### 데이터셋 특성
- **원본 데이터**: Give Me Some Credit (GMSC) 데이터셋
- **샘플 크기**: 150,000명
- **부도율** (2년 내 90일 이상 연체): **6.68%** (정상 93.32%)
- **클래스 분포**: 불균형 (Positive:Negative = 1:14)
- **씬파일러**: 약 4.22% (금융 이력 부족)

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
│   │   ├── cs-training.csv   # Give Me Some Credit 훈련 데이터 (git 포함)
│   │   ├── cs-test.csv       # 테스트 데이터 (git 포함)
│   │   └── Data Dictionary.xls
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
│   │   ├── main.py      # FastAPI 앱 (화면별 태그, Swagger /docs)
│   │   ├── routers/     # 화면별 엔드포인트 (health / predict / explain / fairness / metrics / anova)
│   │   ├── schemas.py   # 요청·응답 스키마 + Swagger 예시
│   │   ├── model.py     # 최종 모델·스케일러·SHAP 로딩, /predict·/explain 계산
│   │   ├── reports.py   # outputs/ 산출물 JSON 읽기 (없으면 examples/ 예시)
│   │   └── examples/    # 산출물이 없을 때 쓰는 예시 JSON (디자인 수치)
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

- API 서버(`src/api/`)는 대시보드 5개 화면이 쓰는 9개 엔드포인트를 갖춤 (`/docs` 에서 화면별 태그로 확인).
  - `POST /predict`, `POST /explain`: `models/` 의 최종 모델·스케일러로 즉시 계산 (모델 파일 없으면 503).
  - `GET /explain/summary`, `/fairness`, `/fairness/mitigation`, `/metrics`, `/anova`: `main_step3~5` 가 `outputs/` 에 저장한 JSON 을 그대로 반환. 파일이 없으면 `src/api/examples/` 의 예시를 반환하며, `GET /health` 의 `reports` 로 어느 쪽인지 알 수 있다.
  - `GET /metrics/comparison`: `models/model_metadata.json` 의 후보 모델 격자.
  - 데이터 반영 방법: 파이프라인을 돌려 `outputs/` 를 채우면 끝. API 코드 수정 불필요.
- Streamlit 대시보드(`src/dashboard/app.py`)는 미션 스펙의 5개 탭(고객심사/XAI분석/공정성/모델성능/통계검증) 구조를 갖추고, API 서버하고만 통신하도록 구성됨 (모델 파일 직접 로드 없음 - MSA 제약 준수).
- `src/data/loader.py`, `src/data/simulator.py`, `src/data/preprocessor.py`는 함수 시그니처만 정의된 상태 - 이재희가 실제 로직 구현 예정.
- `src/models/`는 3가지 모델 학습 (Logistic Regression, XGBoost, LightGBM) - 이재희가 구현 예정.
- `src/statistics/anova.py`, `src/fairness/`, `src/explainer/`는 함수 시그니처만 정의된 상태 - 남궁명진이 실제 로직 구현 예정.
- GMSC 데이터셋(`data/raw/`)이 추가됨. 포함 파일: `cs-training.csv`(훈련 데이터), `cs-test.csv`(테스트 데이터), `sampleEntry.csv`(샘플 제출), `Data Dictionary.xls`(데이터 사전).

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
docker compose up --build     # .env 없이도 뜬다. 처음 한 번은 이미지 빌드에 몇 분 걸림
```

- api: http://localhost:8000 (Swagger: /docs)
- dashboard: http://localhost:8501 — 컨테이너 안에서는 `API_BASE_URL=http://api:8000` 으로 API 를 찾는다
- mlflow: http://localhost:5000 — 이 PC 의 `./mlruns` 기록을 보여 준다. 팀 서버(`mlflow.gosky.kr`) 기록은 `.env` 의 `MLFLOW_TRACKING_URI` 로 학습 스크립트가 올리는 곳이며, 로컬 UI 에서 보려면 `MLFLOW_TRACKING_URI=http://localhost:5000` 으로 학습을 다시 돌린다

포트를 바꾸려면 `.env` 에 `API_PORT`, `DASHBOARD_PORT`, `MLFLOW_PORT` 를 적는다. 대시보드를 실제 API 로 연결하려면 `USE_MOCK=false`.

## 환경 변수 설명

`.env.example` 참고. 실제 실행 시 `.env`로 복사해서 사용.

| 변수 | 설명 |
|---|---|
| DATA_DIR | 데이터 디렉토리 경로 |
| RANDOM_SEED | 재현성을 위한 랜덤 시드 |
| THIN_FILER_RATIO | 시뮬레이터의 씬파일러 비율 조정 파라미터 |
| BIAS_RATIO | 시뮬레이터의 성별/연령대별 승인율 편향 조정 파라미터 |
| API_PORT | FastAPI 서버 포트 (docker compose 호스트 포트, 기본 8000) |
| DASHBOARD_PORT | Streamlit 대시보드 포트 (기본 8501) |
| MLFLOW_PORT | MLflow UI 포트 (기본 5000) |
| API_BASE_URL | 대시보드가 호출할 API 주소 (로컬 `http://localhost:8000`, compose 에서는 자동으로 `http://api:8000`) |
| USE_MOCK | `true` 면 대시보드가 API 대신 mock JSON 을 쓴다 (기본 true) |
| MLFLOW_TRACKING_URI / USERNAME / PASSWORD | 학습 스크립트가 실험을 올릴 MLflow 서버. 비밀번호는 `.env` 에만 적는다 |
