"""데이터 드리프트 표 (모델 성능 화면, 보너스 4, #140).

데이터: GET /drift (#149 결과, API #139). 시나리오(열) × 입력 정보(행) 의 CSI 와 맨 위 점수 PSI 를 한 표에 둔다.
판정(안정/주의/경고)은 API 의 grade 를 그대로 쓰고, 색과 함께 글자로도 보여 준다.
"""
import re
from html import escape
from pathlib import Path

import streamlit as st

import api_client
import features as F
from components import ui
from components.api_error import render_api_error
from components.empty_state import pending_html, render_no_report

CSS_PATH = Path(__file__).with_suffix(".css")
GRADE_KIND = {"안정": "ok", "주의": "mid", "경고": "fail"}
SECTION = "데이터 드리프트 (PSI · CSI)"


def _name(scenario):
    """'1) 경기 악화' → '경기 악화', '0) 기준 확인 (Test 그대로)' → '기준 확인'"""
    name = re.sub(r"^\d+\)\s*", "", scenario.get("name") or scenario.get("description") or "")
    return re.sub(r"\s*\(.*\)$", "", name)


def _num(v):
    return "-" if v is None else f"{v:.1f}" if v >= 10 else f"{v:.3f}"


def _cell(value, grade):
    kind = GRADE_KIND.get(grade, "gray")
    tag = f'<small>{escape(grade)}</small>' if grade and kind != "ok" else ""
    return f'<span class="dr-cell {kind}" title="{escape(grade or "")}">{_num(value)}{tag}</span>'


def _table(data):
    scenarios = list((data.get("scenarios") or {}).values())
    features = list((scenarios[0].get("csi") or {}).keys()) if scenarios else []
    cols = len(scenarios)
    head = "".join(f'<span class="num">{escape(_name(s))}</span>' for s in scenarios)
    score = "".join(_cell(s.get("score_psi"), s.get("score_grade")) for s in scenarios)
    rows = "".join(
        f'<div class="dr-row"><span class="dr-name">{escape(F.label(f))}'
        f'{"<i>대안</i>" if F.is_alternative(f) else ""}</span>'
        + "".join(_cell((s["csi"].get(f) or {}).get("index"), (s["csi"].get(f) or {}).get("grade")) for s in scenarios)
        + "</div>" for f in features)
    extra = [("AUC", lambda s: f'{s["auc"]:.3f}' if s.get("auc") is not None else "-"),
             ("승인율", lambda s: f'{s["approval_rate"] * 100:.1f}%' if s.get("approval_rate") is not None else "-"),
             ("씬파일러 비율", lambda s: f'{s["thin_filer_share"] * 100:.1f}%' if s.get("thin_filer_share") is not None else "-")]
    foot = "".join(f'<div class="dr-row dr-result"><span class="dr-name">{label}</span>'
                   + "".join(f'<span class="num">{fn(s)}</span>' for s in scenarios) + "</div>" for label, fn in extra)
    return f"""
    <div class="dr-table" role="table" aria-label="시나리오별 PSI · CSI" style="--dr-cols:{cols}">
      <div class="dr-row dr-headrow"><span>입력 정보</span>{head}</div>
      <div class="dr-row dr-score"><span class="dr-name">모델 점수 <small>PSI</small></span>{score}</div>
      {rows}
      <div class="dr-group">시나리오를 적용했을 때의 결과</div>
      {foot}
    </div>"""


def _findings(data):
    """주의 이상인 칸마다 가장 많이 바뀐 구간 한 줄."""
    lines = []
    for s in (data.get("scenarios") or {}).values():
        for f in s.get("flagged") or []:
            b = (s.get("top_bins") or {}).get(f) or {}
            grade = (s["csi"].get(f) or {}).get("grade")
            where = (f'"{escape(b["label"])}" 구간 비율 {b["expected"] * 100:.1f}% → {b["actual"] * 100:.1f}%'
                     if b.get("label") is not None else "")
            lines.append(f'<li><b>{escape(_name(s))}</b> · {escape(F.label(f))} '
                         f'{ui.pill(grade or "-", GRADE_KIND.get(grade, "gray"))}<span class="dr-where">{where}</span></li>')
    if not lines:
        return '<p class="dr-none">주의 이상으로 바뀐 입력 정보가 없어요.</p>'
    return f'<ul class="dr-findings">{"".join(lines)}</ul>'


def render():
    with st.container(key="bonus-drift", gap="small"):
        _render()


def _render():
    th = {"stable": 0.1, "caution": 0.25}
    try:
        data = api_client.drift()
    except Exception as e:
        st.html(ui.section_head(SECTION))
        render_api_error("GET /drift", e)
        return
    if isinstance(data, dict):
        th = {**th, **(data.get("thresholds") or {})}
    sub = (f"학습 데이터(Train)와 분포를 비교해요. {th['stable']:g} 미만 안정, {th['stable']:g}~{th['caution']:g} 주의, "
           f"{th['caution']:g} 이상 경고예요. 기준 확인 외 3개는 Test 데이터를 일부러 바꾼 시나리오예요.")
    st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>" + ui.section_head(SECTION, sub))
    if data == api_client.NOT_READY:
        st.html(pending_html("GET /drift", "#139"))
        return
    if data is None:
        render_no_report("drift")
        return
    legend = "".join(f'<span><i class="{k}"></i>{g}</span>' for g, k in GRADE_KIND.items())
    st.html(f'<section class="dr-card">{_table(data)}'
            f'<div class="dr-side"><span class="dr-side-title">가장 많이 바뀐 구간</span>{_findings(data)}'
            f'<div class="dr-legend">{legend}</div>'
            f'<p class="dr-note">PSI는 모델 점수(부도 확률), CSI는 입력 정보 하나하나의 분포 변화예요. '
            f'점수 PSI가 안정이어도 CSI가 경고면 그 정보의 수집 방식이 바뀌었는지 먼저 확인해요.</p></div></section>')
