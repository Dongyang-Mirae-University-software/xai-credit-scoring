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
    assert f['MonthlyIncome'] == CUSTOMER_EXAMPLE['MonthlyIncome'] / model.INCOME_KRW_PER_UNIT  # 원 → 모델 단위
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


# ---------- 대시보드 시안용 추가 (이슈 #116) ----------
def test_screenings_keep_latest_per_customer(tmp_path, monkeypatch):
    monkeypatch.setenv('SCREENINGS_DB', str(tmp_path / 's.db'))

    def save(cid, grade):
        r = client.post('/screenings', json={'input': {'customer_id': cid}, 'result': {
            'approval_status': 'APPROVED', 'risk_grade': grade, 'credit_score': 700}})
        assert r.status_code == 201 and r.json()['screening_id'].startswith('S')
    save('C1', 'B'); save('C2', 'C'); save('C1', 'A')
    rows = client.get('/screenings?limit=5').json()
    assert [(r['customer_id'], r['risk_grade']) for r in rows] == [('C1', 'A'), ('C2', 'C')]
    assert len(client.get('/screenings?limit=1').json()) == 1


def test_customers_lookup_uses_predict_units():
    c = client.get('/customers/C100001')
    assert c.status_code == 200, c.text
    body = c.json()
    assert body['customer_id'] == 'C100001' and 0 <= body['telecom_payment_rate'] <= 100
    assert client.get('/customers/NOPE').status_code == 404
    if model.ready():  # 조회 결과를 그대로 /predict 에 넣을 수 있어야 한다
        assert client.post('/predict', json=body).status_code == 200


def test_new_report_fields():
    s = client.get('/explain/summary').json()
    assert all('higher_is_riskier' in f for f in s['features'])
    assert {'shap', 'value'} == set(next(iter(s['beeswarm'].values())))
    assert {'auc', 'roc_curve', 'pr_curve'} <= set(client.get('/metrics').json()['baseline'])
    assert 'evaluated_at' in client.get('/fairness').json()
    assert 'evaluated_at' in client.get('/fairness/mitigation').json()
    assert client.get('/anova').json()['source']['cv'] == {'n_splits': 5, 'n_repeats': 1}


def test_income_is_won_outside_the_api():
    c = client.get('/customers/C100001').json()
    assert c['MonthlyIncome'] == 9120 * model.INCOME_KRW_PER_UNIT  # 파일은 모델 단위, 응답은 원
    if model.ready():
        r = client.post('/explain', json=c).json()
        income = [x for x in r['rejection_reasons'] if x['feature'] == 'MonthlyIncome']
        assert all(x['current_value'] == c['MonthlyIncome'] for x in income)


def test_metrics_has_cv_thin_filer_uplift():
    # 씬파일러 향상 판정은 Train+Val 교차검증 값으로 한다 (#110, #120)
    m = client.get('/metrics').json()
    cv = m['thin_filer_uplift_cv']
    assert cv['ci_low'] <= cv['uplift'] <= cv['ci_high']
    assert m['pass']['thin_filer_uplift'] == cv['pass']
