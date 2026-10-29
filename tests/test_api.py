"""API 스모크 테스트: 모든 엔드포인트가 스키마에 맞는 응답을 돌려주는지 (모델 파일이 있으면 실제 예측까지)."""
import pytest
from fastapi.testclient import TestClient

from src.api import model
from src.api.main import app
from src.api.schemas import CUSTOMER_EXAMPLE

client = TestClient(app)


def test_health_lists_report_sources():
    body = client.get('/health').json()
    assert body['status'] in ('healthy', 'degraded')
    assert body['disclaimer']
    assert set(body['reports']) == {'metrics', 'shap_summary', 'reference_values', 'fairness', 'fairness_mitigation', 'anova'}


@pytest.mark.parametrize('path, keys', [
    ('/explain/summary', {'features', 'n_samples'}),
    ('/fairness', {'di_ratio', 'by_attribute', 'criteria'}),
    ('/fairness/mitigation', {'before', 'after', 'method'}),
    ('/metrics', {'auc', 'ks', 'confusion_matrix', 'roc_curve', 'pass'}),
    ('/metrics/comparison', {'selected_model', 'val_table', 'cv_runs'}),
    ('/anova', {'test_1', 'test_2'}),
])
def test_report_endpoints(path, keys):
    r = client.get(path)
    assert r.status_code == 200, r.text
    assert keys <= set(r.json())


def test_predict_rejects_out_of_range():
    assert client.post('/predict', json={**CUSTOMER_EXAMPLE, 'telecom_payment_rate': 150}).status_code == 422


def test_percent_inputs_are_divided_by_100():
    """이슈 #70: 부채비율·납부율은 % 로 받고 모델에는 0~1 로 넣는다"""
    from src.api.schemas import CustomerInput
    f = CustomerInput(**{**CUSTOMER_EXAMPLE, 'DebtRatio': 35, 'telecom_payment_rate': 78, 'utility_payment_rate': 100}).features()
    assert (f['DebtRatio'], f['telecom_payment_rate'], f['utility_payment_rate']) == (0.35, 0.78, 1.0)
    assert f['spending_consistency'] == CUSTOMER_EXAMPLE['spending_consistency']  # 점수는 그대로


@pytest.mark.skipif(not model.ready(), reason='models/ 에 최종 모델 파일이 없음')
def test_predict_and_explain_with_real_model():
    p = client.post('/predict', json=CUSTOMER_EXAMPLE)
    assert p.status_code == 200, p.text
    body = p.json()
    assert 0 <= body['credit_score'] <= 1000 and body['risk_grade'] in 'ABCDE'
    assert body['approval_status'] == ('APPROVED' if body['default_probability'] < body['approval_threshold'] else 'REJECTED')

    e = client.post('/explain', json=CUSTOMER_EXAMPLE)
    assert e.status_code == 200, e.text
    body = e.json()
    assert set(body['shap_values']) == set(k for k in CUSTOMER_EXAMPLE if k != 'customer_id')
    for reason in body['rejection_reasons']:
        assert reason['shap_contribution'] < 0 and reason['explanation']


def test_thin_filer_rule():
    thin = {**CUSTOMER_EXAMPLE, 'NumberOfOpenCreditLinesAndLoans': 1, 'NumberRealEstateLoansOrLines': 0,
            'NumberOfTime30-59DaysPastDueNotWorse': 0}
    features = {k: v for k, v in thin.items() if k != 'customer_id'}
    assert model.is_thin_filer(features) is True
    assert model.is_thin_filer({k: v for k, v in CUSTOMER_EXAMPLE.items() if k != 'customer_id'}) is False
