import json

import mlflow
from mlflow.tracking import MlflowClient

from scripts import seed_mlflow as s


def test_build_records_has_final_with_auc_ks():
    meta = json.loads(s.META.read_text(encoding='utf-8'))
    recs = s.build_records(meta)
    assert len(recs) >= 3
    name, _, m = recs[-1]
    assert name.startswith('FINAL integrated/LightGBM/is_unbalance')
    assert {'test_auc', 'test_ks'} <= set(m)


def test_seed_is_idempotent(tmp_path):
    mlflow.set_tracking_uri(tmp_path.as_uri())  # 파일 저장소: venv 의 sqlalchemy 버전과 무관
    c = MlflowClient()
    recs = s.build_records(json.loads(s.META.read_text(encoding='utf-8')))
    s.seed(c, recs)
    s.seed(c, recs)  # 두 번 돌려도 run 이 늘지 않는다
    exp = c.get_experiment_by_name(s.EXPERIMENT)
    assert len(c.search_runs([exp.experiment_id])) == len(recs)
