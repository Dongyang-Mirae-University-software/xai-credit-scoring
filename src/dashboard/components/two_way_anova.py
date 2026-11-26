"""고객 유형 × 모델 구성 Two-way ANOVA (통계 검증 화면, 보너스 3, #140).

데이터: GET /anova/two-way (#148 결과). 결론 문장 → 교호작용 그림(고객 유형별 평균 AUC) + ANOVA 표 → 폴드 대응 확인.
교호작용이 유의하면 '대안 데이터를 넣었을 때 AUC 가 오르는 폭이 고객 유형마다 다르다'는 뜻이다.
"""
import math
from html import escape
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

import api_client
from components import ui
from components.api_error import render_api_error
from components.empty_state import pending_html, render_no_report

CSS_PATH = Path(__file__).with_suffix(".css")
TITLE = "검증 3. 고객 유형 × 모델 구성 (Two-way ANOVA)"
QUESTION = "대안 데이터의 효과가 씬파일러에게 더 클까?"
SOURCES = [("segment", "고객 유형", "씬파일러 / 일반"), ("config", "모델 구성", "전통 / 통합"),
           ("segment×config", "교호작용", "고객 유형 × 모델 구성")]
CONFIG_LABELS = {"전통": "전통 모델", "통합": "통합 모델"}
SEGMENT_STYLE = {"씬파일러": ("#1F4FD8", "solid"), "일반": ("#8A94A6", "dash")}
INK, GRID, MUTED = "#0F1B2D", "#EDF1F6", "#5B6577"


def _p(p):
    return "&lt; 0.001" if p < 0.001 else f"{p:.3f}"


def _p_text(p):
    """문장 안의 p값: 'p &lt; 0.001' 또는 'p = 0.012'."""
    return "p &lt; 0.001" if p < 0.001 else f"p = {p:.3f}"


def _conclusion(data):
    rows = {r["source"]: r for r in data["anova"]["table"]}
    inter = rows.get("segment×config") or {}
    paired = data.get("paired_checks") or {}
    up = paired.get("simple_effects") or {}
    did = paired.get("interaction_did") or {}
    ups = ", ".join(f'{"일반 고객" if s == "일반" else s} {v["mean"]:+.3f}' for s, v in up.items())
    if inter.get("is_significant"):
        bigger = "씬파일러" if did.get("mean", 0) > 0 else "일반 고객"
        head = f"대안 데이터의 효과가 {bigger}에게 더 커요."
    else:
        head = "대안 데이터의 효과가 두 고객 유형에서 비슷해요."
    tail = f" 통합 모델로 바꿨을 때 AUC 향상은 {ups}이에요" if ups else ""
    tail += f" (교호작용 {_p_text(inter['p_value'])})." if inter.get("p_value") is not None else "."
    return f'<p class="stt2-conclusion"><strong>결론.</strong> {head}{tail}</p>'


def _figure(data):
    a = data["anova"]
    means, std, n = a["cell_means"], a.get("cell_std") or {}, a.get("n_per_cell") or 0
    configs = list((a.get("main_means") or {}).get("config", {}).keys()) or ["전통", "통합"]
    segments = list((a.get("main_means") or {}).get("segment", {}).keys()) or ["씬파일러", "일반"]
    fig = go.Figure()
    x = [CONFIG_LABELS.get(c, c) for c in configs]
    for seg in segments:
        color, dash = SEGMENT_STYLE.get(seg, (MUTED, "solid"))
        y = [means.get(f"{seg}|{c}") for c in configs]
        err = [1.96 * std[f"{seg}|{c}"] / math.sqrt(n) if n and f"{seg}|{c}" in std else 0 for c in configs]
        name = "일반 고객" if seg == "일반" else seg
        fig.add_scatter(x=x, y=y, mode="lines+markers+text", name=name,
                        line=dict(color=color, width=3, dash=dash), marker=dict(size=10, color=color),
                        error_y=dict(type="data", array=err, color=color, thickness=1.5, width=6),
                        text=[f"{v:.3f}" for v in y], textposition=["middle left"] + ["middle right"] * (len(y) - 1),
                        textfont=dict(size=13, color=INK),
                        hovertemplate=f"{name}<br>%{{x}} 평균 AUC %{{y:.3f}}<extra></extra>")
    fig.update_layout(
        template=None, height=320, margin=dict(l=56, r=24, t=8, b=40), paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Noto Sans KR, system-ui, sans-serif", color=INK), hoverlabel=dict(font_family="Noto Sans KR"),
        legend=dict(orientation="h", x=0, y=1.08, font=dict(size=12, color=MUTED)),
        xaxis=dict(fixedrange=True, showgrid=False, range=[-0.45, len(configs) - 0.55], linecolor="#CDD5E0", tickfont=dict(size=13, color=MUTED)),
        yaxis=dict(fixedrange=True, gridcolor=GRID, zeroline=False, tickformat=".2f", tickfont=dict(size=12, color=MUTED),
                   title=dict(text="폴드 평균 AUC", font=dict(size=12, color=MUTED))),
    )
    return fig


def _table(data):
    rows = {r["source"]: r for r in data["anova"]["table"]}
    body = "".join(
        f'<div class="stt2-row"><span class="stt2-name"><strong>{label}</strong><small>{sub}</small></span>'
        f'<span class="num">{r["f_statistic"]:.1f}</span><span class="num">{_p(r["p_value"])}</span>'
        f'<span class="num">{r["eta_squared"]:.2f}</span>'
        f'<span class="num">{ui.pill("유의함" if r["is_significant"] else "차이 없음", "ok" if r["is_significant"] else "gray")}</span></div>'
        for key, label, sub in SOURCES if (r := rows.get(key)))
    did = (data.get("paired_checks") or {}).get("interaction_did") or {}
    check = ""
    if did:
        check = (f'<p class="stt2-check"><b>폴드 대응 확인</b> 같은 폴드의 네 칸은 서로 독립이 아니어서, 폴드마다 '
                 f'(씬파일러 향상 - 일반 향상)을 구해 다시 검정했어요: 평균 {did["mean"]:+.3f} '
                 f'(95% CI {did["ci_low"]:.3f}~{did["ci_high"]:.3f}, {_p_text(did["p_value"])}), '
                 f'{did["n_folds"]}개 폴드 중 {did["positive_folds"]}개에서 씬파일러 향상이 더 컸어요.</p>')
    return (f'<div role="table" aria-label="Two-way ANOVA 결과"><div class="stt2-row stt2-head"><span>요인</span>'
            f'<span class="num">F값</span><span class="num">p값</span><span class="num">η²</span>'
            f'<span class="num">결과</span></div>{body}</div>{check}')


def render():
    with st.container(key="bonus-2way", gap="small"):
        _render()


def _render():
    st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>" + ui.section_head(TITLE, QUESTION))
    try:
        data = api_client.two_way_anova()
    except Exception as e:
        render_api_error("GET /anova/two-way", e)
        return
    if data == api_client.NOT_READY:
        st.html(pending_html("GET /anova/two-way", "API 이슈 필요"))
        return
    if data is None:
        render_no_report("two_way_anova")
        return
    design = data.get("design") or {}
    n = data["anova"].get("n_per_cell")
    foot = " · ".join(escape(str(v)) for v in (design.get("data"), design.get("cv"),
                                               f"칸마다 폴드 AUC {n}개" if n else None) if v)
    with st.container(key="stt2-card", gap=None):
        st.html(_conclusion(data))
        left, right = st.columns([1, 1.1], gap=None)
        with left, st.container(key="stt2-plot"):
            st.html('<span class="stt2-col-title">고객 유형별 평균 AUC (95% 신뢰구간)</span>')
            st.plotly_chart(_figure(data), theme=None, config={"displayModeBar": False}, use_container_width=True)
            st.html('<span class="stt2-note">두 선의 기울기가 다를수록 교호작용이 커요.</span>')
        with right, st.container(key="stt2-table"):
            st.html('<span class="stt2-col-title">분산분석 표</span>' + _table(data))
        st.html(f'<div class="stt2-foot">관측값: {foot}</div>')
