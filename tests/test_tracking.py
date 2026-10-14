"""MLflow 쉬운 영어 설명(src/tracking/notes.py), REST 업로드(src/tracking/rest.py), main_step6 테스트"""
import io
import json
import re

import pytest

import main_step6
from src.tracking import notes
from src.tracking.rest import MlflowRest

HANGUL = re.compile('[가-힣]')

PERF = {'auc': 0.984, 'ks': 0.878, 'psi': 0.0006, 'precision': 0.55, 'recall': 0.93,
        'thin_filer_uplift': {'n_thin_filer': 625, 'n_thin_filer_default': 42, 'auc_without_alt': 0.807,
                              'auc_with_alt': 0.975, 'uplift': 0.168, 'ci_low': 0.121, 'ci_high': 0.220}}
FAIR = {'di_ratio': 1.10, 'equalized_odds_tpr_diff': 0.045, 'equalized_odds_fpr_diff': 0.062,
        'by_attribute': {'age_group': {'protected_group': '55+', 'di_ratio': 1.10,
                                       'approval_rates': {'20-34': 0.796, '55+': 0.92}}}}
MITI = {'method': 'threshold_adjustment', 'threshold_fit_on': 'validation',
        'before': {'di_ratio': 1.10, 'equalized_odds_tpr_diff': 0.045, 'equalized_odds_fpr_diff': 0.062},
        'after': {'di_ratio': 0.986, 'equalized_odds_tpr_diff': 0.039, 'equalized_odds_fpr_diff': 0.048}}
ANOVA = {'test_1': {'group_means': {'전통모델': 0.86, '대안모델': 0.97, '통합모델': 0.98}, 'f_statistic': 145.3,
                    'p_value': 1e-9, 'effect_size_eta_squared': 0.77,
                    'post_hoc_tukey': [{'comparison': '전통 vs 통합', 'mean_diff': -0.12, 'p_adj': 0.001, 'significant': True}]},
         'test_2': {'group_means': {'XGBoost': 0.98, 'LightGBM': 0.98}, 'f_statistic': 0.2, 'p_value': 0.66,
                    'effect_size_eta_squared': 0.02, 'post_hoc_tukey': []}}
EXPLAIN = {'rejection_reasons': [{'rank': 1, 'feature': 'telecom_payment_rate', 'current_value': 0.84,
                                  'reference_value': 0.9, 'shap_contribution': -1.67}]}


def test_explain_key_patterns():
    assert 'fold 3 of 5' in notes.explain_key('fold_3_roc_auc')
    assert 'Average of the 5 folds' in notes.explain_key('ks_mean')
    assert 'Small = stable' in notes.explain_key('roc_auc_std')
    assert 'age' in notes.explain_key('psi_age')
    assert notes.explain_key('thin_ci_low').startswith('Lower end')
    assert notes.explain_key('something_else') is None


def test_model_run_note_and_tags():
    tags = notes.model_run_tags('LightGBM', 'is_unbalance', 'integrated')
    assert tags == {'model_family': 'LightGBM', 'data_used': 'financial + alternative data',
                    'imbalance_fix': 'give rare default cases more weight'}
    text = notes.model_run_note('LightGBM', 'is_unbalance', 'integrated',
                                {'cv_test_roc_auc_mean': 0.984, 'roc_auc_std': 0.001, 'cv_test_recall_mean': 0.93})
    assert 'LightGBM model trained on financial + alternative data' in text
    assert '0.984' in text and 'recall 0.930' in text


def test_final_test_note_verdicts():
    text = notes.final_test_note('integrated/LightGBM/is_unbalance', 0.984, 0.878, psi=0.0006,
                                 uplift={'uplift': 0.168, 'ci_low': 0.121, 'ci_high': 0.22})
    assert text.count('passed') == 4 and 'NOT passed' not in text
    assert 'LightGBM, is_unbalance' in text
    low = notes.final_test_note('integrated/LightGBM/is_unbalance', 0.70, 0.20)
    assert low.count('NOT passed') == 2


def test_summary_is_plain_english():
    md = notes.summary_md(PERF, EXPLAIN, FAIR, MITI, ANOVA)
    assert not HANGUL.search(md)                       # 한글 집단 이름도 영어로 바뀜
    for part in ('Model performance', 'Fairness', 'one-way ANOVA', 'financial only vs financial + alternative',
                 'phone bill on-time rate', 'chosen on validation data', notes.DISCLAIMER_EN):
        assert part in md
    assert 'No clear difference' in md and 'really different' in md


def test_metrics_guide_lists_known_keys():
    md = notes.metrics_guide_md()
    assert '`test_auc`' in md and '`psi_telecom_payment_rate`' in md and '`bias_ratio`' in md
    assert not HANGUL.search(md)


class FakeServer:
    """요청을 기록하고 정해 둔 응답을 돌려주는 가짜 MLflow 서버."""
    def __init__(self):
        self.requests = []

    def __call__(self, req):
        body = req.data
        self.requests.append((req.get_method(), req.full_url, dict(req.header_items()), body))
        url = req.full_url
        if 'get-by-name' in url:
            out = {'experiment': {'experiment_id': '1'}}
        elif url.endswith('/runs/search'):
            out = {'runs': [{'info': {'run_id': 'abc', 'run_name': 'final_test',
                                      'artifact_uri': 'mlflow-artifacts:/1/abc/artifacts'},
                             'data': {'metrics': [{'key': 'test_auc', 'value': 0.984}, {'key': 'test_ks', 'value': 0.878}],
                                      'tags': [{'key': 'selected_model', 'value': 'integrated/LightGBM/is_unbalance'}]}}]}
        else:
            out = {}
        return io.BytesIO(json.dumps(out).encode())


def test_rest_upload_url_and_auth(tmp_path):
    f = tmp_path / 'summary_en.md'
    f.write_text('hi', encoding='utf-8')
    server = FakeServer()
    client = MlflowRest('https://example.test/', 'user', 'pw', opener=server)
    rel = client.upload('mlflow-artifacts:/1/abc/artifacts', f, 'evaluation')
    method, url, headers, body = server.requests[-1]
    assert method == 'PUT' and rel == '1/abc/artifacts/evaluation/summary_en.md'
    assert url == 'https://example.test/api/2.0/mlflow-artifacts/artifacts/1/abc/artifacts/evaluation/summary_en.md'
    assert headers['Authorization'].startswith('Basic ') and body == b'hi'
    with pytest.raises(RuntimeError):
        client.upload('s3://bucket/x', f)


def _outputs(tmp_path):
    ev, fa, st = tmp_path / 'evaluation', tmp_path / 'fairness', tmp_path / 'statistics'
    (ev / 'shap').mkdir(parents=True)
    fa.mkdir()
    st.mkdir()
    (ev / 'metrics_model.json').write_text(json.dumps(PERF), encoding='utf-8')
    (ev / 'explain_example.json').write_text(json.dumps(EXPLAIN), encoding='utf-8')
    (ev / 'shap' / 'summary_model.png').write_bytes(b'png')
    (fa / 'fairness.json').write_text(json.dumps(FAIR), encoding='utf-8')
    (fa / 'fairness_mitigation.json').write_text(json.dumps(MITI), encoding='utf-8')
    (st / 'anova.json').write_text(json.dumps(ANOVA, ensure_ascii=False), encoding='utf-8')
    return ev, fa, st


def test_step6_build_and_upload(tmp_path):
    ev, fa, st = _outputs(tmp_path)
    plan, _ = main_step6.build(ev, fa, st, tmp_path / 'summary')
    folders = {(p.name, d) for p, d in plan}
    assert ('summary_en.md', '') in folders and ('summary_model.png', 'evaluation/shap') in folders
    assert ('fairness_mitigation.json', 'fairness') in folders and ('anova.json', 'statistics') in folders
    server = FakeServer()
    rid = main_step6.upload(plan, tmp_path / 'none.env', client=MlflowRest('https://example.test', opener=server))
    assert rid == 'abc'
    puts = [r for r in server.requests if r[0] == 'PUT']
    assert len(puts) == len(plan)
    tags = [json.loads(r[3]) for r in server.requests if r[1].endswith('/runs/set-tag')]
    note = next(t['value'] for t in tags if t['key'] == 'mlflow.note.content')
    assert 'tested once on unseen test data' in note and 'NOT passed' not in note


def test_step6_dry_run_does_not_upload(tmp_path, monkeypatch):
    ev, fa, st = _outputs(tmp_path)
    called = []
    monkeypatch.setattr(main_step6, 'upload', lambda *a, **k: called.append(1))
    main_step6.main(['--eval-dir', str(ev), '--fairness-dir', str(fa), '--stats-dir', str(st), '--out', str(tmp_path / 'o')])
    assert not called and (tmp_path / 'o' / 'summary_en.md').exists()
