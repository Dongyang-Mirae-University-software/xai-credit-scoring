"""모델 성능 페이지 (디자인 07)

결론(성능 목표 4개 달성 여부) → 근거(ROC · PR 곡선) → 상세(혼동행렬 + 해석).
데이터: GET /metrics. 기준 모델(Logistic Regression) 곡선은 /metrics 의 baseline 이 있을 때만 그린다.
"""
from html import escape
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

import api_client
from components import ui
from components.api_error import render_api_error
from components.empty_state import render_no_report
from components.page_header import render_page_header

CSS_PATH = Path(__file__).with_suffix(".css")
DATASET_LABEL = "테스트셋"
DEFAULT_TARGETS = {"auc": 0.78, "ks": 0.28, "thin_filer_uplift": 0.03, "psi": 0.1}
ALGORITHM_LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "logistic": "Logistic Regression"}
ACCENT, BASELINE, INK, GRID, MUTED = "#1F4FD8", "#8A94A6", "#0F1B2D", "#EDF1F6", "#5B6577"
SUBTITLE = "최종 모델이 부도 고객을 얼마나 잘 가려내는지 확인해요."


def _algorithm(model_name):
    name = (model_name or "").lower()
    return next((label for key, label in ALGORITHM_LABELS.items() if name.startswith(key)), model_name or "최종 모델")


def _model_version(model_name):
    return (model_name or "—").removesuffix(".joblib")


# ---------------------------------------------------------------------------
# 결론
# ---------------------------------------------------------------------------
def _uplift(m):
    """씬파일러 향상: 판정 기준인 Train+Val 교차검증 값(#110)을 먼저 쓰고, 없으면 Test 1회 값."""
    return (m.get("thin_filer_uplift_cv") or m.get("thin_filer_uplift") or {}).get("uplift")


def _targets(m):
    t = {**DEFAULT_TARGETS, **(m.get("targets") or {})}
    passes = m.get("pass") or {}
    uplift = _uplift(m)
    items = [
        ("auc", "AUC-ROC", m.get("auc"), f"{m.get('auc', 0):.3f}", f"목표 {t['auc']:.2f} 이상",
         passes.get("auc", m.get("auc", 0) >= t["auc"])),
        ("ks", "KS 통계량", m.get("ks"), f"{m.get('ks', 0):.3f}", f"목표 {t['ks']:.2f} 이상",
         passes.get("ks", m.get("ks", 0) >= t["ks"])),
        ("thin_filer_uplift", "씬파일러 AUC 향상", uplift, "—" if uplift is None else f"{uplift:+.3f}",
         f"목표 +{t['thin_filer_uplift']:.2f} 이상",
         passes.get("thin_filer_uplift", uplift is not None and uplift >= t["thin_filer_uplift"])),
        ("psi", "PSI", m.get("psi"), "—" if m.get("psi") is None else f"{m['psi']:.3f}", f"목표 {t['psi']:g} 미만",
         passes.get("psi", m.get("psi") is not None and m["psi"] < t["psi"])),
    ]
    return t, items


def _summary(m):
    t, items = _targets(m)
    done = [i for i in items if i[5]]
    all_ok = len(done) == len(items)
    auc, uplift = m.get("auc"), _uplift(m)
    first = f"{DATASET_LABEL} AUC {auc:.3f}로 목표({t['auc']:.2f})를 {'넘었' if auc >= t['auc'] else '넘지 못했'}"
    if uplift is None:
        sub = first + "어요."
    elif uplift >= t["thin_filer_uplift"]:
        sub = f"{first}고, 대안 데이터 덕분에 씬파일러 고객의 예측력이 {uplift:.3f} 올랐어요."
    else:
        sub = f"{first}지만, 씬파일러 고객의 AUC 향상은 {uplift:+.3f}로 목표(+{t['thin_filer_uplift']:.2f})에 못 미쳐요."
    title = (f"성능 목표 {len(items)}개를 모두 달성했어요" if all_ok
             else f"성능 목표 {len(items)}개 중 {len(items) - len(done)}개가 미달이에요")
    cells = "".join(
        f'<div class="pf-metric"><span>{label}</span><div><strong>{text}</strong>'
        f'{ui.pill("달성" if ok else "미달", "ok" if ok else "fail")}</div><small>{rule}</small></div>'
        for _, label, _, text, rule, ok in items)
    secondary = [
        ("Precision", m.get("precision"), "거절한 고객 중 실제 부도 고객 비율"),
        ("Recall", m.get("recall"), "실제 부도 고객 중 거절로 걸러낸 비율"),
        ("F1", m.get("f1"), "Precision과 Recall의 조화 평균"),
    ]
    sec = "".join(f'<div><span>{label}</span><strong>{"—" if v is None else f"{v:.3f}"}</strong><small>{desc}</small></div>'
                  for label, v, desc in secondary)
    tone = "ok" if all_ok else "fail"
    return f"""
    <section class="pf-summary">
      <div class="pf-banner {tone}"><span class="pf-mark">{ui.icon("check" if all_ok else "x", "#FFFFFF", 22)}</span>
        <div><strong>{title}</strong><span>{escape(sub)}</span></div></div>
      <div class="pf-metrics">{cells}</div>
      <div class="pf-secondary">{sec}</div>
    </section>"""


# ---------------------------------------------------------------------------
# 근거: ROC · PR (Plotly)
# ---------------------------------------------------------------------------
def _layout(fig, x_title, y_title):
    axis = dict(tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], tickformat=".1f", showgrid=True, gridcolor=GRID, zeroline=False,
                linecolor="#CDD5E0", showline=True, ticks="", tickfont=dict(size=13, color=MUTED),
                title_font=dict(size=14, color="#3A4456"), fixedrange=True)
    fig.update_layout(
        template=None, height=460, margin=dict(l=64, r=12, t=8, b=56), showlegend=False,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Noto Sans KR, system-ui, sans-serif", color=INK),
        xaxis=dict(axis, title_text=x_title, range=[-0.01, 1.02]),
        yaxis=dict(axis, title_text=y_title, range=[-0.01, 1.03]),
        hoverlabel=dict(font_family="Noto Sans KR"),
    )
    return fig


def _roc(m, baseline, name):
    cm = m.get("confusion_matrix") or {}
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=BASELINE, width=1, dash="dash"), hoverinfo="skip")
    if baseline and baseline.get("roc_curve"):
        fig.add_scatter(x=baseline["roc_curve"]["fpr"], y=baseline["roc_curve"]["tpr"], mode="lines",
                        line=dict(color=BASELINE, width=2), name="Logistic Regression",
                        hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra>기준 모델</extra>")
    roc = m.get("roc_curve") or {}
    fig.add_scatter(x=roc.get("fpr", []), y=roc.get("tpr", []), mode="lines", line=dict(color=ACCENT, width=3),
                    name=name, hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra>최종 모델</extra>")
    if cm.get("fp") is not None and (cm["fp"] + cm["tn"]):
        fig.add_scatter(x=[cm["fp"] / (cm["fp"] + cm["tn"])], y=[cm.get("recall", m.get("recall"))], mode="markers",
                        marker=dict(size=13, color="#FFFFFF", line=dict(color=INK, width=2)),
                        hovertemplate="현재 승인 기준점<br>FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>")
    return _layout(fig, "잘못 거절한 정상 고객 비율 (FPR)", "걸러낸 부도 고객 비율 (TPR)")


def _pr(m, baseline, name, default_rate):
    fig = go.Figure()
    if default_rate is not None:
        fig.add_scatter(x=[0, 1], y=[default_rate, default_rate], mode="lines",
                        line=dict(color=BASELINE, width=1, dash="dash"), hoverinfo="skip")
        fig.add_annotation(x=0.98, y=default_rate, yshift=12, xanchor="right", showarrow=False,
                           text=f"무작위 기준선 (부도율 {default_rate * 100:.1f}%)", font=dict(size=12, color=MUTED))
    if baseline and baseline.get("pr_curve"):
        fig.add_scatter(x=baseline["pr_curve"]["recall"], y=baseline["pr_curve"]["precision"], mode="lines",
                        line=dict(color=BASELINE, width=2),
                        hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra>기준 모델</extra>")
    pr = m.get("pr_curve") or {}
    fig.add_scatter(x=pr.get("recall", []), y=pr.get("precision", []), mode="lines", line=dict(color=ACCENT, width=3),
                    hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra>최종 모델</extra>")
    if m.get("recall") is not None and m.get("precision") is not None:
        fig.add_scatter(x=[m["recall"]], y=[m["precision"]], mode="markers",
                        marker=dict(size=13, color="#FFFFFF", line=dict(color=INK, width=2)),
                        hovertemplate="현재 승인 기준점<br>Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>")
    return _layout(fig, "걸러낸 부도 고객 비율 (Recall)", "거절 중 실제 부도 비율 (Precision)")


def _legend(name, baseline):
    base = ('<span><i class="line base"></i>Logistic Regression (기준 모델)</span>' if baseline else "")
    return (f'<div class="pf-legend"><span><i class="line"></i>{escape(name)} (최종 모델)</span>{base}'
            '<span><i class="dot"></i>현재 승인 기준점</span></div>')


# ---------------------------------------------------------------------------
# 상세: 혼동행렬
# ---------------------------------------------------------------------------
def _cell(value, rate, label):
    alpha = 0.12 + 0.56 * rate
    dark = alpha > 0.4
    return (f'<div class="pf-cell{" dark" if dark else ""}" style="background:rgba(31,79,216,{alpha:.2f})">'
            f'<strong>{value:,}</strong><span>{label} ({rate * 100:.1f}%)</span></div>')


def _confusion(m):
    cm = m.get("confusion_matrix") or {}
    tn, fp, fn, tp = (cm.get(k) for k in ("tn", "fp", "fn", "tp"))
    if None in (tn, fp, fn, tp):
        return '<div class="pf-empty">혼동행렬 정보가 없어요.</div>'
    normal, default = tn + fp, fn + tp
    total = normal + default
    approved = tn + fn
    r = lambda a, b: a / b if b else 0  # noqa: E731
    fp_rate = r(fp, normal)
    one_in = round(1 / fp_rate) if fp_rate > 0 else None
    fp_text = (f"정상 고객 {one_in}명 중 1명꼴로 거절돼요. " if one_in and one_in > 1 else "") + \
        "승인 기준을 높이면 줄일 수 있지만 놓치는 부도 고객이 늘어나요."
    grid = f"""
    <div class="pf-cm" role="group" aria-label="혼동행렬">
      <span></span><span class="pf-cm-col">예측: 승인</span><span class="pf-cm-col">예측: 거절</span>
      <span class="pf-cm-row">실제 정상<br>{normal:,}명</span>
      {_cell(tn, r(tn, normal), "맞게 승인")}{_cell(fp, fp_rate, "잘못 거절")}
      <span class="pf-cm-row">실제 부도<br>{default:,}명</span>
      {_cell(fn, r(fn, default), "놓친 부도")}{_cell(tp, r(tp, default), "맞게 거절")}
    </div>"""
    notes = [
        (f"부도 고객 {default:,}명 중 {tp:,}명을 거절했어요",
         f"Recall {r(tp, default) * 100:.1f}%. 놓친 부도 고객은 {fn:,}명이에요."),
        (f"정상 고객 {normal:,}명 중 {fp:,}명을 거절했어요", fp_text),
        (f"전체 승인율은 {r(approved, total) * 100:.1f}%예요", f"{total:,}명 중 {approved:,}명을 승인했어요."),
    ]
    side = "".join(f'<div><strong>{a}</strong><span>{b}</span></div>' for a, b in notes)
    return f'<div class="pf-cm-card">{grid}<div class="pf-cm-notes">{side}</div></div>'


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------
st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")


def _main():
    try:
        metrics = api_client.metrics()
    except Exception as e:
        render_page_header("모델 성능", SUBTITLE)
        render_api_error("GET /metrics", e, "page")
        return
    if metrics is None:
        render_page_header("모델 성능", SUBTITLE)
        render_no_report("metrics")
        return

    n = metrics.get("n")
    render_page_header("모델 성능", SUBTITLE, meta=[("평가 데이터", f"{DATASET_LABEL} {n:,}명" if n else "—"),
                                                 ("모델", _model_version(metrics.get("model")))])
    st.html(_summary(metrics))

    name = _algorithm(metrics.get("model"))
    baseline = metrics.get("baseline")  # {"model", "auc", "roc_curve", "pr_curve"}. 예전 서버엔 없다
    default_rate = metrics["n_default"] / metrics["n"] if metrics.get("n") and metrics.get("n_default") is not None else None

    st.html(ui.section_head("ROC 곡선과 Precision-Recall 곡선",
                            "곡선이 왼쪽 위(ROC), 오른쪽 위(PR)에 가까울수록 부도 고객을 잘 가려내는 모델이에요."))
    left, right = st.columns(2, gap="medium")
    config = {"displayModeBar": False, "staticPlot": False}
    with left, st.container(key="pf-roc"):
        base_auc = f' (기준 모델 {baseline["auc"]:.3f})' if baseline and baseline.get("auc") is not None else ""
        st.html(f'<div class="pf-card-head"><h3>ROC Curve</h3>'
                f'<span>AUC <strong>{metrics["auc"]:.3f}</strong>{base_auc}</span></div>')
        st.plotly_chart(_roc(metrics, baseline, name), theme=None, config=config, use_container_width=True)
        st.html(_legend(name, baseline))
    with right, st.container(key="pf-pr"):
        rate_text = f"부도율 {default_rate * 100:.1f}%의 불균형 데이터" if default_rate is not None else ""
        st.html(f'<div class="pf-card-head"><h3>Precision-Recall Curve</h3><span>{rate_text}</span></div>')
        st.plotly_chart(_pr(metrics, baseline, name, default_rate), theme=None, config=config, use_container_width=True)
        st.html(_legend(name, baseline))

    threshold = (metrics.get("confusion_matrix") or {}).get("threshold")
    st.html(f'<section class="pf-section">'
            + ui.section_head("혼동행렬", f"{DATASET_LABEL} {n:,}명을 현재 승인 기준"
                              + (f"(부도 확률 {threshold:.0%} 미만 승인)" if threshold is not None else "")
                              + "으로 심사했을 때의 결과예요. 색이 진할수록 해당 집단에서 차지하는 비율이 커요.")
            + _confusion(metrics) + "</section>")


_main()
