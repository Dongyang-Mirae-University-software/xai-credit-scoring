# MLflow 트래킹 서버

팀 전체가 실험을 기록하는 MLflow 서버 하나를 띄우는 폴더. 이 폴더만 서버에 복사해도 배포된다.

| 파일 | 역할 |
|---|---|
| `docker-compose.yml` | MLflow 서버 (SQLite 백엔드 + 아티팩트 서빙, Model Registry 사용 가능) |
| `.env.example` | 포트 · 컨테이너 이름 |
| `nginx.conf.template` | 외부 공개용 리버스 프록시 + basic auth |
| `mlruns/` | 실행하면 생김. 실험 DB(`mlflow.db`) + 아티팩트 전부. **git 제외, 백업 대상** |

## 로컬에서 쓰기

```bash
cd mlflow
cp .env.example .env
docker compose up -d
open http://127.0.0.1:5000     # macOS 는 localhost 대신 127.0.0.1 (AirPlay 가 5000 을 가로챈다)
```

## 서버 배포 (Ubuntu + nginx)

MLflow 에는 인증이 없다. **nginx basic auth 없이 인터넷에 열지 말 것** — 누구나 실험·모델을 지울 수 있다.

**1. 기동**

```bash
cd mlflow
cp .env.example .env            # MLFLOW_PORT 를 서버에서 비어 있는 포트로 (ss -tlnH 로 확인)
docker compose up -d
docker compose ps               # healthy 확인
curl -s http://127.0.0.1:$(grep ^MLFLOW_PORT .env | cut -d= -f2)/health    # OK
```

> docker 데이터 경로가 NTFS/exFAT 디스크여도 문제없다 — `mlruns/` 는 이 폴더에 bind mount 된다.

**2. 비밀번호 파일**

```bash
sudo apt install -y apache2-utils
PW=$(openssl rand -base64 18); echo "$PW"      # 팀에 공유할 비밀번호. 따로 보관
sudo htpasswd -bc /etc/nginx/.htpasswd-mlflow <아이디> "$PW"
```

**3. nginx**

```bash
DOMAIN=mlflow.example.com
PORT=$(grep ^MLFLOW_PORT .env | cut -d= -f2)
sed -e "s/__DOMAIN__/$DOMAIN/" -e "s/__MLFLOW_PORT__/$PORT/" nginx.conf.template \
  | sudo tee /etc/nginx/sites-available/$DOMAIN >/dev/null
sudo ln -sf /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
curl -s -o /dev/null -w "%{http_code}\n" -H "Host: $DOMAIN" http://127.0.0.1/    # 401 이면 정상
```

**4. HTTPS** — DNS A 레코드가 서버 IP 를 가리킨 뒤

```bash
sudo certbot --nginx -d $DOMAIN --redirect
```

## 기록하기 (클라이언트)

학습 코드는 환경 변수만 바꾸면 된다. 코드에 주소·비밀번호를 쓰지 않는다.

```bash
export MLFLOW_TRACKING_URI=https://mlflow.example.com
export MLFLOW_TRACKING_USERNAME=<아이디>
export MLFLOW_TRACKING_PASSWORD=<비밀번호>
python -m src.models.train ...
```

클라이언트 `mlflow` 버전은 서버 이미지(2.22)와 맞춘다 (`pip install "mlflow>=2.22,<2.23"`). 어긋나면 Registry·아티팩트 API 가 깨질 수 있다.

로깅 규약 (반드시 지킬 것):

- 실험 이름 `xai-credit-scoring`, 등록 모델 이름 `xai-credit-model`
- run 1개 = (알고리즘 × 불균형 기법 × 피처 그룹) 조합 1개. 이름은 `lr-class_weight-traditional` 형식
- **폴드별 AUC 를 전부** `mlflow.log_metric("auc_fold", v, step=i)` 로 남긴다 — ANOVA 가 이 값을 되읽는다. 평균만 남기면 나중에 전부 재학습해야 한다
- 결과를 MLflow 밖(엑셀 등)에 따로 기록하지 않는다
- 대시보드는 MLflow 에서 모델을 직접 로드하지 않는다 (실격 조항). 모델 로드는 API 에서만

## 운영

```bash
docker compose logs -f                 # 로그
docker compose pull && docker compose up -d   # 이미지 교체 (버전 올릴 땐 클라이언트도 같이)
tar czf mlruns-$(date +%F).tgz mlruns  # 백업 — 이 폴더가 곧 전체 데이터
```

기존 서버의 기록을 옮길 때: 기존 `mlruns/`(안에 `mlflow.db` · `mlartifacts/`)를 이 폴더에 복사한 뒤 `docker compose up -d`.
