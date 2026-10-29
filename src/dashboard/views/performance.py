"""모델 성능 페이지"""
import base64
from html import escape
from pathlib import Path

import streamlit as st

import api_client
from components.page_header import render_page_header

CSS_PATH = Path(__file__).with_suffix(".css")

ALGORITHMS = [
    ("LogisticRegression", "Logistic Regression"),
    ("XGBoost", "XGBoost"),
    ("LightGBM", "LightGBM"),
]
DATA_TYPES = [
    ("financial_only", "전통 only"),
    ("alternative_only", "대안 only"),
    ("integrated", "통합 (전통+대안)"),
]
DATA_TYPE_LABELS = {"financial_only": "전통 피처", "alternative_only": "대안 피처", "integrated": "통합 피처"}
DATASET_LABEL = "테스트셋"  # 백엔드 main_step3.py 는 Test 데이터로 /metrics 재료를 만든다


# ---------------------------------------------------------------------------
# 그림 도우미 (SVG, 0~100 좌표)
# ---------------------------------------------------------------------------
def _polyline(xs, ys):
    """x·y (0~1) 점 목록 → SVG points. y는 위가 1이 되도록 뒤집는다."""
    return " ".join(f"{x * 100:.2f},{(1 - y) * 100:.2f}" for x, y in zip(xs, ys))


LINE_COLORS = {"roc": "#2563eb", "pr": "#7c3aed"}


def _chart(line_points, line_class, dashed_points):
    """곡선 SVG를 이미지로 넣는다 (st.html 은 보안상 <svg> 태그를 지우므로 data URI <img> 로 전달)."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" preserveAspectRatio="none">'
        f'<polyline points="{dashed_points}" fill="none" stroke="#9ca3af" stroke-width="1" '
        'stroke-dasharray="4 4" vector-effect="non-scaling-stroke"/>'
        f'<polyline points="{line_points}" fill="none" stroke="{LINE_COLORS[line_class]}" stroke-width="2.5" '
        'stroke-linejoin="round" vector-effect="non-scaling-stroke"/></svg>'
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img class="pf-plot" alt="{line_class} curve" src="data:image/svg+xml;base64,{encoded}">'


# ---------------------------------------------------------------------------
# 화면 조각
# ---------------------------------------------------------------------------
def _metric_cards(m):
    targets, passed = m.get("targets", {}), m.get("pass", {})
    threshold = m["confusion_matrix"]["threshold"]

    def badge(key, ok_label, ng_label):
        if key not in passed:
            return '<span class="pf-pill none">—</span>'
        return (f'<span class="pf-pill ok">{ok_label}</span>' if passed[key]
                else f'<span class="pf-pill ng">{ng_label}</span>')

    cards = [
        ("AUC", m["auc"], 3, badge("auc", "통과", "미달"), f'기준 ≥ {targets.get("auc", 0.78):.2f} · {DATASET_LABEL}'),
        ("KS", m["ks"], 2, badge("ks", "통과", "미달"), f'기준 ≥ {targets.get("ks", 0.28):.2f} · 누적분포 최대차'),
        ("PSI", m.get("psi"), 3, badge("psi", "안정", "불안정"), f'기준 &lt; {targets.get("psi", 0.1):.2f} · train vs test'),
        ("Precision", m["precision"], 2, badge(None, "", ""), f"임계값 {threshold:.2f}"),
        ("Recall", m["recall"], 2, badge(None, "", ""), f"임계값 {threshold:.2f}"),
        ("F1", m["f1"], 2, badge(None, "", ""), f"임계값 {threshold:.2f}"),
    ]
    html = "".join(
        f'<div class="pf-card pf-metric">'
        f'<div class="pf-metric-head"><span>{name}</span>{pill}</div>'
        f'<div class="pf-metric-value">{"—" if value is None else f"{value:.{digits}f}"}</div>'
        f'<div class="pf-caption">{caption}</div></div>'
        for name, value, digits, pill, caption in cards
    )
    st.html(f'<div class="pf-metrics">{html}</div>')


def _roc_card(m):
    roc = m["roc_curve"]
    return f"""
    <div class="pf-card">
      <div class="pf-title">ROC Curve</div>
      <div class="pf-caption">GET /metrics · AUC = {m["auc"]:.3f} · 점선 = 무작위 (0.5)</div>
      {_chart(_polyline(roc["fpr"], roc["tpr"]), "roc", "0,100 100,0")}
      <div class="pf-axis"><span>FPR →</span><span>TPR ↑</span></div>
    </div>"""


def _pr_card(m):
    pr = m["pr_curve"]
    base = m["n_default"] / m["n"] if m.get("n") else 0
    return f"""
    <div class="pf-card">
      <div class="pf-title">Precision-Recall Curve</div>
      <div class="pf-caption">양성 비율 {base:.2%} · 점선 = 기준선 ({base:.3f})</div>
      {_chart(_polyline(pr["recall"], pr["precision"]), "pr", f"0,{(1 - base) * 100:.2f} 100,{(1 - base) * 100:.2f}")}
      <div class="pf-axis"><span>Recall →</span><span>Precision ↑</span></div>
    </div>"""


def _confusion_card(m):
    cm = m["confusion_matrix"]
    return f"""
    <div class="pf-card">
      <div class="pf-title">혼동행렬</div>
      <div class="pf-caption">{DATASET_LABEL} {m["n"]:,}건 · 임계값 {cm["threshold"]:.2f} · 양성 = 부도</div>
      <div class="pf-cm">
        <span></span><span class="pf-cm-col">예측: 정상</span><span class="pf-cm-col">예측: 부도</span>
        <span class="pf-cm-row">실제: 정상</span>
        <div class="pf-cm-cell tn"><b>TN</b>{cm["tn"]:,}</div>
        <div class="pf-cm-cell fp"><b>FP</b>{cm["fp"]:,}</div>
        <span class="pf-cm-row">실제: 부도</span>
        <div class="pf-cm-cell fn"><b>FN</b>{cm["fn"]:,}</div>
        <div class="pf-cm-cell tp"><b>TP</b>{cm["tp"]:,}</div>
      </div>
      <div class="pf-chips">
        <span class="pf-chip">Recall {cm["recall"]:.2f} = TP/(TP+FN)</span>
        <span class="pf-chip">Precision {cm["precision"]:.2f} = TP/(TP+FP)</span>
      </div>
    </div>"""


def _model_grid_card(runs_data, m, stage):
    """runs_data: GET /metrics/comparison 응답 (selected_model, cv_runs = 후보별 5-Fold CV 평균)."""
    runs = list(runs_data.get("cv_runs", {}).values())
    data_type, algorithm, _ = (runs_data["selected_model"].split("/") + ["", "", ""])[:3]

    def base_run(algo, dtype):
        # 알고리즘마다 SMOTE 가 아닌 기본 불균형 기법(class_weight 계열) 결과를 쓴다
        for r in runs:
            if r["algorithm"] == algo and r["data_type"] == dtype and r["method"] != "SMOTE":
                return r
        return None

    def cell(r):
        if not r:
            return "—"
        mean = r["cv_test_mean"]["cv_test_roc_auc_mean"]
        std = r.get("cv_test_std", {}).get("cv_test_roc_auc_std")
        return f"{mean:.3f} ± {std:.3f}" if std is not None else f"{mean:.3f}"

    rows = []
    for algo, label in ALGORITHMS:
        selected = algo == algorithm
        cells = "".join(
            f'<td class="{"pf-sel" if selected and dtype == data_type else ""}">{cell(base_run(algo, dtype))}</td>'
            for dtype, _ in DATA_TYPES)
        integrated = base_run(algo, "integrated")
        ks = f'{integrated["cv_test_mean"]["cv_test_ks_mean"]:.2f}' if integrated else "—"
        psi = f'{m["psi"]:.3f}' if selected and m.get("psi") is not None else "—"  # PSI 는 최종 모델만 계산됨
        rows.append(f'<tr><th class="{"pf-sel" if selected else ""}">{label}</th>{cells}<td>{ks}</td><td>{psi}</td></tr>')

    pick = f"최종 선택: {escape(algorithm)} · {DATA_TYPE_LABELS.get(data_type, data_type)}"
    if stage:
        pick += f" · Registry {escape(stage)}"
    head = "".join(f"<th>{label}</th>" for _, label in DATA_TYPES)
    return f"""
    <div class="pf-card">
      <div class="pf-card-head">
        <div>
          <div class="pf-title">모델 격자 비교 · AUC (mean ± sd, 5-Fold)</div>
          <div class="pf-caption">MLflow 실험 {len(runs)}조합 · 불균형 기법: class_weight 계열(SMOTE 제외) 기준 표기</div>
        </div>
        <span class="pf-chip blue">{pick}</span>
      </div>
      <table class="pf-table">
        <thead><tr><th>알고리즘</th>{head}<th>KS</th><th>PSI</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>"""


def _thin_filer_card(m):
    t = m.get("thin_filer_uplift")
    if not t:
        return """
        <div class="pf-card">
          <div class="pf-title">씬파일러 세그먼트</div>
          <div class="pf-empty">/metrics 응답에 thin_filer_uplift 가 없습니다.</div>
        </div>"""
    note = ("대안 데이터가 씬파일러 판별력을 높임" if t["pass"]
            else f"목표(+{t.get('target', 0.03):.2f}) 미달")
    ci = f' · 95% CI [{t["ci_low"]:+.2f}, {t["ci_high"]:+.2f}]' if "ci_low" in t else ""
    return f"""
    <div class="pf-card">
      <div class="pf-title">씬파일러 세그먼트</div>
      <div class="pf-caption">대안 데이터 유 / 무 · 씬파일러 {t["n_thin_filer"]:,}명 ({DATASET_LABEL})</div>
      <div class="pf-thin">
        <div class="pf-thin-row"><span>대안 데이터 포함</span><b class="ok">AUC {t["auc_with_alt"]:.2f}</b></div>
        <div class="pf-thin-row"><span>전통 데이터만</span><b>AUC {t["auc_without_alt"]:.2f}</b></div>
      </div>
      <span class="pf-pill {"ok" if t["pass"] else "ng"} wide">{t["uplift"]:+.2f} AUC · {note}{ci}</span>
    </div>"""


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------
st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")
render_page_header("모델 성능", "ROC · PR 곡선과 혼동행렬, AUC · KS · PSI 지표로 최종 모델을 검증합니다")

try:
    metrics = api_client.metrics()
except Exception as e:
    st.error(f"/metrics 호출 실패: {e}")
    st.stop()

_metric_cards(metrics)
st.html(f'<div class="pf-row3">{_roc_card(metrics)}{_pr_card(metrics)}{_confusion_card(metrics)}</div>')

try:
    runs_data = api_client.model_comparison()
    stage = api_client.health().get("model_stage")
    grid = _model_grid_card(runs_data, metrics, stage)
except Exception as e:
    grid = f'<div class="pf-card"><div class="pf-title">모델 격자 비교</div><div class="pf-empty">불러오지 못했습니다: {escape(str(e))}</div></div>'
st.html(f'<div class="pf-row2">{grid}{_thin_filer_card(metrics)}</div>')
