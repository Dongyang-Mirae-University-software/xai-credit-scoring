"""API 스모크 테스트: 모든 엔드포인트가 스키마에 맞는 응답을 돌려주는지 (모델 파일이 있으면 실제 예측까지)."""
import pytest
from fastapi.testclient import TestClient

from src.api import model, reports
from src.api.main import app
from src.api.schemas import CUSTOMER_EXAMPLE

client = TestClient(app)


def test_health_lists_report_sources():
    body = client.get('/health').json()
    assert body['status'] in ('healthy', 'degraded')
    assert body['disclaimer']
    assert set(body['reports']) == {'metrics', 'shap_summary', 'reference_values', 'fairness', 'fairness_mitigation', 'anova', 'drift'}


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
    assert model.SCORE_MIN <= body['credit_score'] <= model.SCORE_MAX and body['risk_grade'] in 'ABCDE'
    assert body['risk_grade'] == model.risk_grade(body['credit_score']) and body['score_method'] == 'pdo-600-50-20'
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


# ---------- PDO 점수 (이슈 #157) ----------
def test_pdo_score_anchor_and_clipping():
    assert model.credit_score(1 / 51) == 600  # 기준점: good:bad 50:1 → 600점
    assert model.credit_score(1 / 101) - model.credit_score(1 / 51) in (19, 20, 21)  # odds 2배 → +PDO(20)점
    assert (model.credit_score(0.0), model.credit_score(1.0)) == (model.SCORE_MAX, model.SCORE_MIN)
    assert model.approval_score() == model.credit_score(model.APPROVAL_THRESHOLD) == 487


def test_grade_boundaries():
    assert [model.risk_grade(s) for s in (300, 462, 463, 486, 487, 504, 505, 526, 527, 850)] == \
        ['E', 'E', 'D', 'D', 'C', 'C', 'B', 'B', 'A', 'A']


def test_health_sends_score_scale_for_dashboard():
    h = client.get('/health').json()
    assert h['grade_scale'] == [list(g) for g in model.GRADE_SCALE] and h['score_range'] == [300, 850]
    assert h['approval_score'] == 487 and h['score_method'] == model.SCORE_METHOD


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
    assert c['MonthlyIncome'] == round(9120 * model.INCOME_KRW_PER_UNIT)  # 파일은 모델 단위, 응답은 원 (원 단위 반올림)
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


def test_drift_tables_only_on_request():
    if reports.load('drift', fallback=False) is None:
        pytest.skip('python main_bonus_drift.py 를 먼저 실행')
    body = client.get('/drift').json()
    assert {'method', 'thresholds', 'scenarios'} <= set(body)
    assert all('tables' not in s for s in body['scenarios'].values())
    full = client.get('/drift', params={'include_tables': 'true'}).json()
    assert all('tables' in s for s in full['scenarios'].values())


def test_drift_default_call_does_not_strip_later_tables():
    if reports.load('drift', fallback=False) is None:
        pytest.skip('python main_bonus_drift.py 를 먼저 실행')
    client.get('/drift')
    full = client.get('/drift', params={'include_tables': 'true'}).json()
    assert all('tables' in s for s in full['scenarios'].values())


def test_drift_missing_file_returns_example(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, 'OUTPUTS_DIR', tmp_path)
    for params in ({}, {'include_tables': 'true'}):
        r = client.get('/drift', params=params)
        assert r.status_code == 200, r.text
        assert r.json()['scenarios'].keys() == {'baseline', 'app_login_change'}
        assert all('tables' not in s for s in r.json()['scenarios'].values())


def test_lookup_by_name_and_rrn():
    """POST /customers/lookup: Swagger 예시 그대로 조회된다. 은행 기록(CB · 대안정보)만 주고 주민등록번호는 없다."""
    from src.api.schemas import LOOKUP_EXAMPLE, CustomerProfile
    r = client.post('/customers/lookup', json=LOOKUP_EXAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['customer_id'] == 'C100001' and body['name'] == LOOKUP_EXAMPLE['name']
    # 고객이 신청서에 적는 값(월소득 · 부양가족 수)과 그걸로 계산하는 부채비율은 주지 않는다
    assert not {'rrn', 'MonthlyIncome', 'NumberOfDependents', 'DebtRatio'} & set(body)
    record = client.get('/customers/C100001').json()
    assert 'rrn' not in record
    assert body['MonthlyRepayment'] == round(record['DebtRatio'] / 100 * record['MonthlyIncome'])
    assert body == CustomerProfile.model_config['json_schema_extra']['examples'][0]   # Swagger 응답 예시 = 실제 응답
    # '-' 없이 보내도, 이름에 공백이 섞여도 같은 고객
    digits = LOOKUP_EXAMPLE['rrn'].replace('-', '')
    assert client.post('/customers/lookup', json={'name': f" {LOOKUP_EXAMPLE['name']} ", 'rrn': digits}).status_code == 200
    if model.ready():  # 조회 결과 + 고객이 적은 월소득 · 부양가족 수 → 부채비율을 만들어 /predict 에 넣을 수 있어요
        income = 4_000_000
        payload = {k: v for k, v in body.items() if k not in ('name', 'gender', 'MonthlyRepayment')}
        payload |= {'MonthlyIncome': income, 'NumberOfDependents': 1,
                    'DebtRatio': body['MonthlyRepayment'] / income * 100}
        assert client.post('/predict', json=payload).status_code == 200


def test_lookup_not_found_and_bad_format():
    from src.api.schemas import LOOKUP_EXAMPLE
    r = client.post('/customers/lookup', json={**LOOKUP_EXAMPLE, 'name': '없는사람'})
    assert r.status_code == 404
    assert r.json()['detail'] != 'Not Found'   # 대시보드는 기본 404('경로 없음')와 '고객 없음'을 이 문구로 구분한다
    assert client.post('/customers/lookup', json={**LOOKUP_EXAMPLE, 'rrn': '0000000000000'}).status_code == 404
    assert client.post('/customers/lookup', json={**LOOKUP_EXAMPLE, 'rrn': '81040821'}).status_code == 422


def test_customer_identity_matches_age_and_gender():
    """가짜 주민등록번호: 기준일 만 나이 = age, 뒷자리 첫 숫자 = 성별, 검증 숫자는 일부러 틀림, 번호는 모두 다름."""
    import json
    from datetime import date
    from scripts.add_customer_identity import PATH, REFERENCE_DATE, check_digit
    customers = json.loads(PATH.read_text(encoding='utf-8'))
    assert len({c['rrn'] for c in customers}) == len(customers)
    for c in customers:
        rrn = c['rrn']
        century = 1900 if rrn[6] in '12' else 2000
        born = date(century + int(rrn[:2]), int(rrn[2:4]), int(rrn[4:6]))
        age = REFERENCE_DATE.year - born.year - ((REFERENCE_DATE.month, REFERENCE_DATE.day) < (born.month, born.day))
        assert age == c['age'], c['customer_id']
        assert (rrn[6] in '13') == (c['gender'] == '남성'), c['customer_id']
        assert int(rrn[12]) != check_digit(rrn[:12]), c['customer_id']
        assert c['name']


def test_latest_screening_by_name_and_rrn(tmp_path, monkeypatch):
    """POST /screenings/lookup: 이름 · 주민등록번호로 그 고객의 가장 최근 심사(입력값 + 결과)를 돌려준다."""
    from src.api.schemas import LOOKUP_EXAMPLE
    monkeypatch.setenv('SCREENINGS_DB', str(tmp_path / 's.db'))
    # 심사 기록이 없으면 404 (고객 없음과 다른 문구)
    r = client.post('/screenings/lookup', json=LOOKUP_EXAMPLE)
    assert r.status_code == 404 and '심사 기록' in r.json()['detail']
    for grade, income in (('B', 3_000_000), ('E', 1_000_000)):   # 같은 고객을 두 번 심사 → 나중 것
        client.post('/screenings', json={'input': {'customer_id': 'C100001', 'MonthlyIncome': income},
                                         'result': {'customer_id': 'C100001', 'approval_status': 'REJECTED',
                                                    'risk_grade': grade, 'credit_score': 100}})
    client.post('/screenings', json={'input': {'customer_id': 'C999999'}, 'result': {'risk_grade': 'A'}})
    body = client.post('/screenings/lookup', json=LOOKUP_EXAMPLE).json()
    assert body['customer_id'] == 'C100001' and body['name'] == LOOKUP_EXAMPLE['name']
    assert body['result']['risk_grade'] == 'E' and body['input']['MonthlyIncome'] == 1_000_000
    assert 'rrn' not in body and 'rrn' not in body['input']
    # 없는 고객은 고객 조회와 같은 404 문구
    r = client.post('/screenings/lookup', json={**LOOKUP_EXAMPLE, 'name': '없는사람'})
    assert r.status_code == 404 and '일치하는 고객' in r.json()['detail']
