import hashlib
import json

import pytest

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
    c = MlflowClient(tracking_uri=tmp_path.as_uri())  # 파일 저장소. 전역 set_tracking_uri 는 다른 테스트로 새어 나간다
    recs = s.build_records(json.loads(s.META.read_text(encoding='utf-8')))
    n, final_id = s.seed(c, recs)
    assert n == len(recs) and c.get_run(final_id).info.run_name.startswith('FINAL')
    s.seed(c, recs)  # 두 번 돌려도 run 이 늘지 않는다
    exp = c.get_experiment_by_name(s.EXPERIMENT)
    assert len(c.search_runs([exp.experiment_id])) == len(recs)


def _file_client(tmp_path):
    uri = tmp_path.as_uri()
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def test_register_staging_is_idempotent(tmp_path):
    c = _file_client(tmp_path)
    recs = s.build_records(json.loads(s.META.read_text(encoding='utf-8')))
    for _ in range(2):  # 두 번 돌려도 버전이 쌓이지 않는다
        _, run_id = s.seed(c, recs)
        s.register(c, run_id)
    versions = c.search_model_versions(f"name='{s.REGISTERED_MODEL}'")
    assert [str(v.version) for v in versions] == ['1']  # 파일 저장소는 int, REST 는 str
    v = versions[0]
    assert v.current_stage == 'Staging' and v.run_id == run_id
    assert str(c.get_model_version_by_alias(s.REGISTERED_MODEL, 'staging').version) == '1'
    assert v.tags['sha256'] == hashlib.sha256(s.MODEL_FILE.read_bytes()).hexdigest()
    assert {a.path for a in c.list_artifacts(run_id, 'model')} == {
        'model/' + s.MODEL_FILE.name, 'model/' + s.SCALER_FILE.name}


def test_register_leaves_hand_made_model(tmp_path):
    c = _file_client(tmp_path)
    c.create_registered_model(s.REGISTERED_MODEL)  # 사람이 직접 등록한 모델 (tag 없음)
    _, run_id = s.seed(c, s.build_records(json.loads(s.META.read_text(encoding='utf-8'))))
    with pytest.raises(SystemExit):
        s.register(c, run_id)
    assert c.get_registered_model(s.REGISTERED_MODEL).tags == {}
