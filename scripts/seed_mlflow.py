"""models/model_metadata.json 의 학습 결과를 MLflow 서버에 run 으로 기록한다.

clone 직후 docker compose up 만 해도 localhost:5000 에 실험 기록이 보이게 하려는 용도다.
실제 학습 run 이 아니라 메타데이터를 옮겨 적은 것이며 (tag source=model_metadata),
실행할 때마다 이전에 옮긴 run 을 지우고 다시 쓰므로 재학습 뒤 메타데이터가 바뀌면 그대로 갱신된다.

사용: MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/seed_mlflow.py
"""
import json
import os
import sys
import time
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

META = Path(__file__).resolve().parents[1] / 'models' / 'model_metadata.json'
EXPERIMENT = 'credit-scoring-metadata'
TAG = 'source'


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
    return len(records)


if __name__ == '__main__':
    client = MlflowClient()
    wait_for_server(client)
    n = seed(client, build_records(json.loads(META.read_text(encoding='utf-8'))))
    print(f'{n} runs -> {mlflow.get_tracking_uri()} / {EXPERIMENT}')
