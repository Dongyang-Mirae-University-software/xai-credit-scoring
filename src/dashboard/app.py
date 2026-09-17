"""
Streamlit 대시보드.
탭: 고객 심사 / XAI 분석 / 공정성 / 모델 성능 / 통계 검증
MSA 제약: 모델 파일을 직접 로드하지 않고, API 서버(src/api/main.py)와
HTTP 통신으로만 동작한다. 현재 API가 mock 응답을 반환하므로 실제 모델
학습이 끝나기 전에도 화면 UI/UX를 먼저 완성할 수 있다.
"""
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="XAI 기반 대안신용평가 시스템", layout="wide")

if "api_base" not in st.session_state:
    st.session_state.api_base = "http://localhost:8000"
if "last_customer" not in st.session_state:
    st.session_state.last_customer = None
if "last_predict" not in st.session_state:
    st.session_state.last_predict = None

with st.sidebar:
    st.header("설정")
    st.session_state.api_base = st.text_input("API 서버 주소", st.session_state.api_base)
    if st.button("서버 상태 확인"):
        try:
            r = requests.get(f"{st.session_state.api_base}/health", timeout=3)
            r.raise_for_status()
            st.success(f"연결됨: {r.json()}")
        except Exception as e:
            st.error(f"연결 실패: {e}")

st.title("XAI 기반 대안신용평가 시스템")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["고객 심사", "XAI 분석", "공정성", "모델 성능", "통계 검증"]
)

API = st.session_state.api_base

# ---------------------------------------------------------------------------
# 탭 1. 고객 심사
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("고객 정보 입력")
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            customer_id = st.text_input("고객 ID", "C000001")
            age = st.number_input("나이", 18, 100, 35)
            monthly_income = st.number_input("월 소득(원)", 0, value=2_500_000, step=100_000)
            debt_ratio = st.slider("부채비율", 0.0, 2.0, 0.4, 0.01)
        with col2:
            revolving_utilization = st.slider("신용한도 소진율", 0.0, 2.0, 0.3, 0.01)
            n_30_59 = st.number_input("30~59일 연체 횟수", 0, value=0)
            n_60_89 = st.number_input("60~89일 연체 횟수", 0, value=0)
            n_90 = st.number_input("90일 이상 연체 횟수", 0, value=0)
        with col3:
            telecom_payment_rate = st.slider("통신비 정상납부율", 0.0, 1.0, 0.9, 0.01)
            utility_payment_rate = st.slider("공과금 납부율", 0.0, 1.0, 0.9, 0.01)
            spending_consistency = st.slider("소비 일관성 점수", 0.0, 100.0, 70.0, 1.0)
            regular_payment_count = st.number_input("정기결제 건수", 0, 20, 5)
            app_login_frequency = st.number_input("앱 로그인 빈도(월)", 0, 30, 10)

        submitted = st.form_submit_button("심사 실행")

    if submitted:
        payload = {
            "customer_id": customer_id,
            "age": age,
            "monthly_income": monthly_income,
            "debt_ratio": debt_ratio,
            "revolving_utilization": revolving_utilization,
            "number_of_time_30_59_days_past_due": n_30_59,
            "number_of_time_60_89_days_past_due": n_60_89,
            "number_of_times_90_days_late": n_90,
            "telecom_payment_rate": telecom_payment_rate,
            "utility_payment_rate": utility_payment_rate,
            "spending_consistency": spending_consistency,
            "regular_payment_count": regular_payment_count,
            "app_login_frequency": app_login_frequency,
        }
        try:
            r = requests.post(f"{API}/predict", json=payload, timeout=5)
            r.raise_for_status()
            result = r.json()
            st.session_state.last_customer = payload
            st.session_state.last_predict = result

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("신용점수", result["credit_score"])
            c2.metric("리스크 등급", result["risk_grade"])
            c3.metric("부도확률", f"{result['default_probability']*100:.1f}%")
            status = result["approval_status"]
            c4.metric("승인 여부", "승인" if status == "APPROVED" else "거절")
            st.caption(f"model_version={result['model_version']} · {result['timestamp']}")
        except Exception as e:
            st.error(f"API 호출 실패: {e}")

    if not submitted and st.session_state.last_predict:
        st.info("가장 최근 심사 결과가 표시되고 있습니다. 새로 심사하려면 위 폼을 제출하세요.")

# ---------------------------------------------------------------------------
# 탭 2. XAI 분석
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("SHAP 기반 심사 근거 분석")
    if not st.session_state.last_customer:
        st.warning("먼저 '고객 심사' 탭에서 심사를 실행하세요.")
    else:
        try:
            r = requests.post(f"{API}/explain", json=st.session_state.last_customer, timeout=5)
            r.raise_for_status()
            explain = r.json()

            st.write(f"승인 여부: **{explain['approval_status']}** "
                     f"· 주요 영향 요인 {explain['total_rejection_factors']}개")

            reasons = explain["rejection_reasons"]
            fig = go.Figure(go.Bar(
                x=[x["shap_contribution"] for x in reasons][::-1],
                y=[x["feature_name_kr"] for x in reasons][::-1],
                orientation="h",
                marker_color=["crimson" if x["shap_contribution"] > 0 else "steelblue" for x in reasons][::-1],
            ))
            fig.update_layout(title="변수별 기여도(SHAP contribution)", height=400)
            st.plotly_chart(fig, use_container_width=True)

            for x in reasons:
                st.markdown(f"**{x['rank']}. {x['feature_name_kr']}** — {x['explanation']}")
        except Exception as e:
            st.error(f"API 호출 실패: {e}")

# ---------------------------------------------------------------------------
# 탭 3. 공정성
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("공정성(Fairness) 지표")
    try:
        r = requests.get(f"{API}/fairness", timeout=5)
        r.raise_for_status()
        fairness = r.json()

        c1, c2, c3 = st.columns(3)
        di = fairness["di_ratio"]
        c1.metric("Disparate Impact Ratio", f"{di:.2f}", "기준 0.8~1.25")
        c2.metric("Equalized Odds - TPR 차이", f"{fairness['equalized_odds_tpr_diff']:.2f}", "기준 0.10 이하")
        c3.metric("Equalized Odds - FPR 차이", f"{fairness['equalized_odds_fpr_diff']:.2f}", "기준 0.10 이하")
        if di < 0.8 or di > 1.25:
            st.warning("DI Ratio가 기준(0.8~1.25)을 벗어났습니다. 편향 완화(Mitigation)가 필요합니다.")
        else:
            st.success("DI Ratio가 기준 범위 내에 있습니다.")

        st.divider()
        st.subheader("편향 완화 전/후 비교")
        r2 = requests.get(f"{API}/fairness/mitigation", timeout=5)
        r2.raise_for_status()
        mitigation = r2.json()
        before, after = mitigation["before"], mitigation["after"]

        metrics_kr = {
            "di_ratio": "DI Ratio",
            "equalized_odds_tpr_diff": "TPR 차이",
            "equalized_odds_fpr_diff": "FPR 차이",
        }
        fig = go.Figure()
        fig.add_trace(go.Bar(name="완화 전", x=list(metrics_kr.values()), y=[before[k] for k in metrics_kr]))
        fig.add_trace(go.Bar(name="완화 후", x=list(metrics_kr.values()), y=[after[k] for k in metrics_kr]))
        fig.update_layout(barmode="group", height=400)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"API 호출 실패: {e}")

# ---------------------------------------------------------------------------
# 탭 4. 모델 성능
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("모델 성능 지표")
    try:
        r = requests.get(f"{API}/metrics", timeout=5)
        r.raise_for_status()
        metrics = r.json()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("AUC", f"{metrics['auc']:.2f}")
        c2.metric("KS", f"{metrics['ks']:.2f}")
        c3.metric("Precision", f"{metrics['precision']:.2f}")
        c4.metric("Recall", f"{metrics['recall']:.2f}")
        c5.metric("F1", f"{metrics['f1']:.2f}")

        fig = go.Figure(go.Bar(x=list(metrics.keys()), y=list(metrics.values())))
        fig.update_layout(title="모델 성능 지표 요약", height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("※ 현재는 mock 값입니다. 실제 모델 학습 완료 후 API가 실측치로 교체됩니다.")
    except Exception as e:
        st.error(f"API 호출 실패: {e}")

# ---------------------------------------------------------------------------
# 탭 5. 통계 검증
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("ANOVA 통계 검증")
    try:
        r = requests.get(f"{API}/anova", timeout=5)
        r.raise_for_status()
        anova = r.json()

        for key, label in [("test_1", "검증 1: 피처 그룹별 비교"), ("test_2", "검증 2: 알고리즘별 비교")]:
            test = anova[key]
            st.markdown(f"### {label}")
            st.write(test["test_name"])

            fig = go.Figure(go.Bar(
                x=list(test["group_means"].keys()),
                y=list(test["group_means"].values()),
            ))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("F-statistic", test["f_statistic"])
            c2.metric("p-value", test["p_value"])
            c3.metric("η² (효과크기)", test["effect_size_eta_squared"])
            c4.metric("유의함", "예" if test["is_significant"] else "아니오")

            if "post_hoc_tukey" in test:
                st.write("Tukey HSD 사후 검정")
                st.table(test["post_hoc_tukey"])
            st.divider()
    except Exception as e:
        st.error(f"API 호출 실패: {e}")
