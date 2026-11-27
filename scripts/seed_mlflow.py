"""models/model_metadata.json 의 학습 결과를 MLflow 서버에 run 으로 기록한다.

clone 직후 docker compose up 만 해도 localhost:5000 에 실험 기록이 보이게 하려는 용도다.
실제 학습 run 이 아니라 메타데이터를 옮겨 적은 것이며 (tag source=model_metadata),
실행할 때마다 이전에 옮긴 run 을 지우고 다시 쓰므로 재학습 뒤 메타데이터가 바뀌면 그대로 갱신된다.
최종 모델 파일은 FINAL run 에 올리고 Model Registry 에 'xai-credit-model' Staging 으로 등록한다 (#165, 팀 서버와 같은 이름).

사용: MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/seed_mlflow.py
"""
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MODELS = Path(__file__).resolve().parents[1] / 'models'
META = MODELS / 'model_metadata.json'
MODEL_FILE = MODELS / 'lightgbm_is_unbalance_integrated_v1.0.joblib'
SCALER_FILE = MODELS / 'preprocess_scaler_v1.0.joblib'
EXPERIMENT = 'credit-scoring-metadata'
TAG = 'source'
REGISTERED_MODEL = 'xai-credit-model'


def build_records(meta):
    """메타데이터 → [(run_name, params, metrics)]. 모델 18개(5-Fold 평균) + 최종 모델 1개."""
    records = []
    for name, r in meta['training_runs']['runs'].items():
        params = {k: r[k] for k in ('data_type', 'algorithm', 'method')}
        metrics = {k: v for k, v in r['cv_test_mean'].items()}
        records.append((name, params, metrics))
    sel = meta['selection']
    final_metrics = {k: sel[k] for k in ('val_auc', 'test_auc', 'test_ks')}
    final_metrics['score_psi'] = sel['psi']['score_psi']
    final_metrics['feature_psi_max'] = sel['psi']['feature_psi']['max']
    if 'feature_csi' in sel['psi']:  # #141 이전 메타데이터에는 없다
        final_metrics['feature_csi_max'] = sel['psi']['feature_csi']['max']
    records.append(('FINAL ' + sel['selected_model'],
                    {'selected_model': sel['selected_model'], 'selection_metric': sel['selection_metric']},
                    final_metrics))
    return records


def wait_for_server(client, tries=30):
    for _ in range(tries):
        try:
            client.search_experiments(max_results=1)
            return
        except Exception:
            time.sleep(2)
    sys.exit('MLflow 서버에 연결할 수 없다: ' + str(mlflow.get_tracking_uri()))


def seed(client, records):
    """run 기록. (run 수, FINAL run id) 를 돌려준다 (FINAL 이 마지막 record)."""
    exp = client.get_experiment_by_name(EXPERIMENT)
    exp_id = exp.experiment_id if exp else client.create_experiment(EXPERIMENT)
    old = client.search_runs([exp_id], f"tags.{TAG} = 'model_metadata'", max_results=1000)
    for r in old:
        client.delete_run(r.info.run_id)
    for name, params, metrics in records:
        run = client.create_run(exp_id, run_name=name, tags={TAG: 'model_metadata'})
        for k, v in params.items():
            client.log_param(run.info.run_id, k, v)
        for k, v in metrics.items():
            client.log_metric(run.info.run_id, k, v)
        client.set_terminated(run.info.run_id)
    return len(records), run.info.run_id


def register(client, run_id):
    """FINAL run 에 모델 파일을 올리고 Registry 에 버전 1 = Staging (alias staging) 으로 등록한다.
    이 스크립트가 만든 모델(tag source=model_metadata)만 지우고 다시 만들어 버전이 쌓이지 않게 한다."""
    try:
        existing = client.get_registered_model(REGISTERED_MODEL)
    except MlflowException as e:  # 파일 저장소는 RestException 이 아니라 MlflowException 을 던진다
        if e.error_code != 'RESOURCE_DOES_NOT_EXIST':
            raise
        existing = None
    if existing is not None:
        if existing.tags.get(TAG) != 'model_metadata':
            sys.exit(f"Registry 에 '{REGISTERED_MODEL}' 이 이미 있고 이 스크립트가 만든 것이 아니다 (tag {TAG} 없음). 건드리지 않고 멈춘다.")
        client.delete_registered_model(REGISTERED_MODEL)

    for f in (MODEL_FILE, SCALER_FILE):
        client.log_artifact(run_id, str(f), artifact_path='model')
    client.create_registered_model(REGISTERED_MODEL, tags={TAG: 'model_metadata'},
                                   description='최종 모델 (models/ 의 파일, scripts/seed_mlflow.py 가 등록)')
    mv = client.create_model_version(
        REGISTERED_MODEL, source=f'{client.get_run(run_id).info.artifact_uri}/model', run_id=run_id,
        tags={'model_file': MODEL_FILE.name, 'sha256': hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest()})
    with warnings.catch_warnings():  # stage 는 2.x 에서 FutureWarning (alias 권장). 과제 기준이 Staging 이라 둘 다 둔다
        warnings.simplefilter('ignore', FutureWarning)
        client.transition_model_version_stage(REGISTERED_MODEL, mv.version, 'Staging')
    client.set_registered_model_alias(REGISTERED_MODEL, 'staging', mv.version)
    return mv.version


if __name__ == '__main__':
    client = MlflowClient()
    wait_for_server(client)
    n, final_run_id = seed(client, build_records(json.loads(META.read_text(encoding='utf-8'))))
    print(f'{n} runs -> {mlflow.get_tracking_uri()} / {EXPERIMENT}')
    version = register(client, final_run_id)
    print(f'{REGISTERED_MODEL} version {version} -> Staging (alias staging)')
