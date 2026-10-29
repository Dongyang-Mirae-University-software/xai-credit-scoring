"""XAI 분석 페이지"""
import math
import re
from html import escape
from pathlib import Path

import streamlit as st

import api_client
import config
from components.page_header import render_page_header

CSS_PATH = Path(__file__).with_suffix(".css")
SUMMARY_MAX_DISPLAY = 15

# 대안 데이터 5개 (미션 문서 정의) — 나머지는 전통 신용 데이터
ALTERNATIVE_FEATURES = {
    "telecom_payment_rate", "utility_payment_rate", "spending_consistency",
    "regular_payment_count", "app_login_frequency",
}
# 화면에 보여줄 한글 변수 이름 (영어 코드 이름은 마우스 오버 시 표시)
# 백엔드 src/explainer/feature_names.py 의 FEATURE_NAMES_KR 과 같게 유지한다.
# (대시보드는 API로만 통신하므로 백엔드 모듈을 직접 불러오지 않고 같은 표를 둔다)
FEATURE_NAMES_KR = {
    "RevolvingUtilizationOfUnsecuredLines": "한도 사용 비율",
    "age": "나이",
    "NumberOfTime30-59DaysPastDueNotWorse": "최근 2년 30~59일 연체 횟수",
    "DebtRatio": "부채 비율",
    "MonthlyIncome": "월소득",
    "NumberOfOpenCreditLinesAndLoans": "보유 대출·신용 한도 개수",
    "NumberOfTimes90DaysLate": "90일 이상 연체 횟수",
    "NumberRealEstateLoansOrLines": "부동산 담보대출 개수",
    "NumberOfTime60-89DaysPastDueNotWorse": "최근 2년 60~89일 연체 횟수",
    "NumberOfDependents": "부양가족 수",
    "telecom_payment_rate": "통신비 정상납부율",
    "utility_payment_rate": "공과금 납부율",
    "spending_consistency": "소비 일관성 점수",
    "regular_payment_count": "정기결제 건수",
    "app_login_frequency": "앱 로그인 빈도",
}
DATASET_LABELS = {"validation": "검증셋", "test": "테스트셋", "train": "학습셋"}
GRADE_RISK = {"A": "low", "B": "low", "C": "mid", "D": "high", "E": "high"}
RISK_LABELS = {"high": "고위험", "mid": "중위험", "low": "저위험"}
# 백엔드 문장 끝의 "(기여도: -0.92)" — 카드에는 기여도를 크게 따로 보여주므로 문장에서는 뺀다
CONTRIBUTION_SUFFIX = re.compile(r"\s*\(기여도:\s*[-+−]?[\d.]+\)\s*$")


# ---------------------------------------------------------------------------
# 표기 도우미
# ---------------------------------------------------------------------------
def _label(feature):
    """한글 변수 이름. 표에 없는 이름은 그대로 보여준다."""
    return FEATURE_NAMES_KR.get(feature, feature)


def _signed(value):
    """+0.92 / −0.18 (마이너스는 유니코드 기호로)."""
    return f"{value:+.2f}".replace("-", "−")


def _num(value):
    """화면에 보여줄 값: 큰 금액은 3.9M, 정수는 그대로, 소수는 둘째 자리까지."""
    if value is None:
        return "—"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:.2f}".rstrip("0").rstrip(".") if abs(value) < 1 else f"{value:,.2f}"


def _age_group(age):
    if age is None:
        return None
    return "20–34" if age < 35 else "35–54" if age < 55 else "55+"


def _risk_of(phi):
    if phi >= config.XAI_RISK_HIGH:
        return "high"
    if phi >= config.XAI_RISK_MID:
        return "mid"
    return "low"


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


# ---------------------------------------------------------------------------
# 상태
# ---------------------------------------------------------------------------
def _init_state():
    st.session_state.setdefault("customers", {})
    st.session_state.setdefault("xai_explain", {})
    if not st.session_state.get("xai_samples_added"):
        _add_sample_customers()
        st.session_state.xai_samples_added = True


def _add_sample_customers():
    """mock 모드: 고객 심사를 거치지 않아도 승인·거절 예시 고객을 고를 수 있게 넣어 둔다."""
    try:
        samples = {
            payload["customer_id"]: {"payload": payload, "result": api_client.predict(payload)}
            for payload in api_client.sample_customers()
        }
    except Exception as e:
        st.warning(f"예시 고객을 불러오지 못했습니다: {e}")
        return
    # 예시 고객은 가장 오래된 쪽에 두어, 고객 심사에서 직접 심사한 고객이 목록 위에 오게 한다
    samples.update(st.session_state.customers)
    st.session_state.customers = samples


def _customer_ids():
    """최근에 심사한 고객이 먼저 오도록."""
    return list(reversed(st.session_state.customers.keys()))


def _get_explain(customer_id, force=False):
    cache = st.session_state.xai_explain
    if force or customer_id not in cache:
        payload = st.session_state.customers[customer_id]["payload"]
        cache[customer_id] = api_client.explain(payload)
    return cache[customer_id]


# ---------------------------------------------------------------------------
# 화면 조각
# ---------------------------------------------------------------------------
def _toolbar(ids):
    """고객 선택 줄. 선택된 고객 ID와 '/explain 다시 요청' 여부를 돌려준다."""
    with st.container(key="xai-toolbar"):
        if not ids:
            msg_col, btn_col = st.columns([5, 1.3], vertical_alignment="center")
            with msg_col:
                st.html('<div class="xai-muted">아직 심사한 고객이 없습니다. '
                        '<b>고객 심사</b>에서 심사를 실행하면 이곳에서 근거를 볼 수 있습니다.</div>')
            with btn_col:
                if st.button("고객 심사로 이동", key="xai-go-screening", use_container_width=True):
                    st.switch_page("views/screening.py")
            return None, False

        if st.session_state.get("xai_customer") not in ids:
            st.session_state.xai_customer = ids[0]

        label_col, select_col, badge_col, chip_col, btn_col = st.columns(
            [0.55, 1.1, 2.3, 3.0, 1.3], vertical_alignment="center")
        with label_col:
            st.html('<div class="xai-label">고객 선택</div>')
        with select_col:
            customer_id = st.selectbox("고객 선택", ids, key="xai_customer", label_visibility="collapsed")

        entry = st.session_state.customers[customer_id]
        result, payload = entry["result"], entry["payload"]
        approved = result["approval_status"] == "APPROVED"
        risk = GRADE_RISK.get(result["risk_grade"], "high")
        with badge_col:
            st.html(f'<div class="xai-chips">'
                    f'<span class="xai-pill {risk}">{"승인" if approved else "거절"} · {result["credit_score"]}점 · '
                    f'등급 {escape(result["risk_grade"])}</span></div>')
        with chip_col:
            group = _age_group(payload.get("age"))
            base_label = f"동일 연령대({group}) 평균" if group else "전체 평균"
            st.html(f'<div class="xai-chips right">'
                    f'<span class="xai-chip">기준값: {base_label}</span>'
                    f'<span class="xai-chip">SHAP 공간: log-odds</span></div>')
        with btn_col:
            refresh = st.button("POST /explain", key="xai-explain", type="primary", use_container_width=True)
    return customer_id, refresh


def _summary_card():
    try:
        summary = api_client.shap_summary()
    except Exception as e:
        st.error(f"전역 중요도를 불러오지 못했습니다: {e}")
        return

    items = sorted(summary["mean_abs_shap"].items(), key=lambda kv: kv[1], reverse=True)[:SUMMARY_MAX_DISPLAY]
    top = items[0][1] if items else 1
    dataset = DATASET_LABELS.get(summary.get("dataset"), summary.get("dataset", ""))
    size = summary.get("sample_size")
    caption = f"mean |SHAP| · 상위 {len(items)}개 변수"
    if size:
        caption += f" · {escape(dataset)} {size:,}건"

    rows = "".join(
        f'<div class="xai-sum-row">'
        f'<span class="xai-feat" title="{escape(f)}">{escape(_label(f))}</span>'
        f'<span class="xai-sum-track"><span class="xai-sum-bar {"alt" if f in ALTERNATIVE_FEATURES else "trad"}" '
        f'style="width:{v / top * 100:.1f}%"></span></span>'
        f'<span class="xai-sum-val">{v:.2f}</span></div>'
        for f, v in items
    )
    st.html(f"""
    <div class="xai-card-head">
      <div>
        <div class="xai-card-title">SHAP Summary · 전역 중요도</div>
        <div class="xai-card-caption">{caption}</div>
      </div>
      <div class="xai-chips">
        <span class="xai-legend trad">■ 전통 데이터</span>
        <span class="xai-legend alt">■ 대안 데이터</span>
      </div>
    </div>
    <div class="xai-sum">{rows}</div>
    """)


def _waterfall_card(customer_id, explain, entry):
    shap_values = explain.get("shap_values") or {}
    base = explain.get("base_value")
    if not shap_values or base is None:
        st.html(f'<div class="xai-card-title">Waterfall · 개별 고객 {escape(customer_id)}</div>'
                '<div class="xai-empty">/explain 응답에 shap_values 또는 base_value가 없어 Waterfall을 그릴 수 없습니다.</div>')
        return

    payload, result = entry["payload"], entry["result"]
    fx = base + sum(shap_values.values())
    p = _sigmoid(fx)
    threshold = config.APPROVAL_THRESHOLD
    decision = "거절" if p >= threshold else "승인"

    ordered = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    shown, rest = ordered[:config.WATERFALL_MAX_DISPLAY], ordered[config.WATERFALL_MAX_DISPLAY:]
    rows = [(f"{_label(f)} = {_num(payload.get(f))}", f, v) for f, v in shown]
    if rest:
        rows.append((f"기타 {len(rest)}개 변수", None, sum(v for _, v in rest)))
    scale = max(abs(v) for _, _, v in rows) or 1

    rows_html = "".join(
        f'<div class="xai-wf-row">'
        f'<span class="xai-feat" title="{escape(feature or label)}">{escape(label)}</span>'
        f'<span class="xai-wf-track"><span class="xai-wf-bar {"up" if v >= 0 else "down"}" '
        f'style="width:{abs(v) / scale * 50:.1f}%"></span></span>'
        f'<span class="xai-wf-val {"up" if v >= 0 else "down"}">{_signed(v)}</span></div>'
        for label, feature, v in rows
    )

    predicted = result.get("default_probability")
    if predicted is not None and abs(p - predicted) <= config.ADDITIVITY_TOLERANCE:
        additivity = '<span class="xai-pill low">가법성 검증: Σφ + base = f(x) ✓</span>'
    elif predicted is not None:
        additivity = (f'<span class="xai-pill mid">가법성 불일치: σ(Σφ + base) = {p:.3f}, '
                      f'/predict = {predicted:.3f}</span>')
    else:
        additivity = ""

    st.html(f"""
    <div class="xai-card-head">
      <div>
        <div class="xai-card-title">Waterfall · 개별 고객 {escape(customer_id)}</div>
        <div class="xai-card-caption">E[f(x)] = {_signed(base)} → f(x) = {_signed(fx)}
          (p = {p:.3f} {"≥" if p >= threshold else "&lt;"} {threshold:.2f} → {decision})</div>
      </div>
    </div>
    <div class="xai-wf">{rows_html}</div>
    <div class="xai-chips">
      <span class="xai-pill high">+ 부도 위험 증가</span>
      <span class="xai-pill blue">− 부도 위험 감소</span>
      {additivity}
    </div>
    """)


def _reasons_card(explain):
    reasons = explain.get("rejection_reasons") or []
    approved = explain.get("approval_status") == "APPROVED"

    head = """
    <div class="xai-card-head">
      <div>
        <div class="xai-card-title">거절 사유 Top 5</div>
        <div class="xai-card-caption">POST /explain → rejection_reasons[] · 항목당 4개 필드: feature · current_value · shap_contribution · explanation</div>
      </div>
      <span class="xai-chip">고객 고지용 문장 · 기준값 = 동일 연령대 평균</span>
    </div>
    """
    if not reasons:
        message = "승인된 고객이라 거절 사유가 없습니다." if approved else "/explain 응답에 거절 사유가 없습니다."
        st.html(head + f'<div class="xai-empty">{message}</div>')
        return

    cards = []
    for r in reasons:
        phi = -r["shap_contribution"]  # 백엔드는 승인 쪽 기준(음수)으로 주므로 부도 쪽 기준으로 뒤집어 표시
        risk = _risk_of(phi)
        sentence = CONTRIBUTION_SUFFIX.sub("", r["explanation"])
        reference = r.get("reference_value")
        ref_html = (f'<span class="xai-reason-ref"><span class="xai-reason-lbl">기준</span> {_num(reference)}</span>'
                    if reference is not None else "")
        # 카드의 다섯 줄(순위 / 이름 / 값 / 기여도 / 문장)은 xai.css 의 subgrid 로 카드끼리 같은 높이에 맞춘다
        cards.append(f"""
        <div class="xai-reason">
          <div class="xai-reason-top">
            <span class="xai-pill {risk}">#{r["rank"]}</span>
            <span class="xai-pill {risk}">{RISK_LABELS[risk]}</span>
          </div>
          <div class="xai-reason-feat" title="{escape(r["feature"])}">{escape(r.get("feature_name_kr") or _label(r["feature"]))}</div>
          <div class="xai-reason-values">
            <span class="xai-reason-cur"><span class="xai-reason-lbl">현재</span> {_num(r["current_value"])}</span>
            {ref_html}
          </div>
          <div class="xai-reason-phi risk-{risk}">{_signed(phi)}<span>log-odds</span></div>
          <div class="xai-reason-text" title="{escape(r["explanation"])}">{escape(sentence)}</div>
        </div>
        """)
    st.html(head + f'<div class="xai-reasons">{"".join(cards)}</div>')
    if not approved and len(reasons) < 3:
        st.warning(f"거절 사유가 {len(reasons)}개입니다. 규제 요건상 거절 시 최소 3개 이상의 사유를 제시해야 합니다.")


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------
st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")
_init_state()
render_page_header("XAI 분석", "SHAP TreeExplainer 로 전역·개별 기여도를 시각화하고, 거절 사유 Top 5 를 자연어로 고지합니다")

customer_id, refresh = _toolbar(_customer_ids())

explain = None
if customer_id:
    try:
        explain = _get_explain(customer_id, force=refresh)
    except Exception as e:
        st.error(f"/explain 호출 실패: {e}")

left, right = st.columns([1, 0.86], gap="medium")
with left:
    with st.container(key="xai-summary"):
        _summary_card()
with right:
    with st.container(key="xai-waterfall"):
        if explain:
            _waterfall_card(customer_id, explain, st.session_state.customers[customer_id])
        else:
            st.html('<div class="xai-card-title">Waterfall · 개별 고객</div>'
                    '<div class="xai-empty">고객을 선택하면 개별 기여도가 표시됩니다.</div>')

with st.container(key="xai-reasons"):
    if explain:
        _reasons_card(explain)
    else:
        st.html('<div class="xai-card-title">거절 사유 Top 5</div>'
                '<div class="xai-empty">고객을 선택하면 거절 사유가 표시됩니다.</div>')
