"""공정성 페이지"""
from html import escape
from pathlib import Path

import streamlit as st

import api_client
from components.api_error import api_error_html, render_api_error
from components.empty_state import no_report_html, render_no_report
from components.page_header import render_page_header

CSS_PATH = Path(__file__).with_suffix(".css")

DATASET_LABEL = "테스트셋"  # 백엔드 main_step4.py 는 Test 데이터로 공정성 지표를 계산한다
ATTRIBUTES = [("gender", "성별"), ("age_group", "연령대")]
GROUP_LABELS = {"male": "남성", "female": "여성"}
BAR_COLORS = {
    "gender": ["#2563eb", "#60a5fa"],
    "age_group": ["#7c3aed", "#a78bfa", "#c4b5fd"],
}
DEFAULT_CRITERIA = {"di_min": 0.8, "di_max": 1.25, "eo_max": 0.1}
DEFAULT_TARGETS = {"auc": 0.78, "ks": 0.28}  # /metrics 의 targets 를 못 받으면 원본 기준값


def _group_label(group):
    return GROUP_LABELS.get(group, group.replace("-", "–"))


def _ordered_groups(attr, rates):
    # 성별은 남성 → 여성, 연령대는 응답 순서(어린 순) 그대로
    order = ["male", "female"] if attr == "gender" else list(rates)
    return [g for g in order if g in rates] + [g for g in rates if g not in order]


def _reference_group(info):
    """DI 의 분모가 되는 기준 집단: 보호집단을 뺀 집단 중 승인율이 가장 높은 집단."""
    others = {g: r for g, r in info["approval_rates"].items() if g != info["protected_group"]}
    return max(others, key=others.get) if others else None


# ---------------------------------------------------------------------------
# 화면 조각
# ---------------------------------------------------------------------------
def _pill(ok, ok_label="양호", ng_label="WARNING"):
    return f'<span class="fa-pill {"ok" if ok else "ng"}">{ok_label if ok else ng_label}</span>'


def _metric_cards(f, criteria):
    by_attr = f["by_attribute"]
    di_range = f'기준 {criteria["di_min"]:.2f} – {criteria["di_max"]:.2f}'
    cards = []
    for attr, label in ATTRIBUTES:
        info = by_attr.get(attr)
        if not info:
            continue
        ref = _reference_group(info)
        caption = f'{di_range} · {_group_label(info["protected_group"])} / {_group_label(ref)} 승인율'
        cards.append((f"DI Ratio · {label}", info["di_ratio"], info["di_passed"], caption))
    eo_max = criteria["eo_max"]
    for key, label, desc in [("equalized_odds_tpr_diff", "TPR", "진양성률"),
                             ("equalized_odds_fpr_diff", "FPR", "위양성률")]:
        cards.append((f"Equalized Odds · {label} 차이", f[key], f[key] <= eo_max,
                      f"기준 ≤ {eo_max:.2f} · 집단 간 {desc} 차이"))

    html = "".join(
        f'<div class="fa-card fa-metric">'
        f'<div class="fa-metric-head"><span>{name}</span>{_pill(ok)}</div>'
        f'<div class="fa-metric-value {"ok" if ok else "ng"}">{value:.2f}</div>'
        f'<div class="fa-caption">{caption}</div></div>'
        for name, value, ok, caption in cards
    )
    st.html(f'<div class="fa-metrics">{html}</div>')


def _approval_card(f, criteria):
    by_attr = f["by_attribute"]
    rates = [r for attr, _ in ATTRIBUTES if attr in by_attr for r in by_attr[attr]["approval_rates"].values()]
    top = max(rates, default=1) or 1

    groups = []
    for attr, _ in ATTRIBUTES:
        info = by_attr.get(attr)
        if not info:
            continue
        bars = "".join(
            f'<div class="fa-bar-col"><div class="fa-bar-area">'
            f'<span class="fa-bar" style="height:{rate / top * 100:.1f}%;background:{BAR_COLORS[attr][i % len(BAR_COLORS[attr])]}">'
            f'<span class="fa-bar-val">{rate:.0%}</span></span></div>'
            f'<span class="fa-bar-label">{escape(_group_label(group))}</span></div>'
            for i, group in enumerate(_ordered_groups(attr, info["approval_rates"]))
            for rate in [info["approval_rates"][group]]
        )
        groups.append(f'<div class="fa-bar-group">{bars}</div>')

    notes = []
    for attr, label in ATTRIBUTES:
        info = by_attr.get(attr)
        if info and not info["di_passed"]:
            bound = "하한" if info["di_ratio"] < criteria["di_min"] else "상한"
            limit = criteria["di_min"] if bound == "하한" else criteria["di_max"]
            notes.append(f'{label} {info["di_ratio"]:.2f} → {bound} {limit:.2f} '
                         f'{"미달" if bound == "하한" else "초과"}로 WARNING')
    note = "DI Ratio = 보호집단 승인율 ÷ 기준집단 승인율. " + (
        " · ".join(notes) + ", Bias Mitigation 대상" if notes else "모든 속성이 기준 안에 있습니다.")

    n = f'{DATASET_LABEL} {f["n"]:,}건 · ' if f.get("n") else ""
    return f"""
    <div class="fa-card">
      <div class="fa-card-head">
        <div>
          <div class="fa-title">집단별 승인율</div>
          <div class="fa-caption">GET /fairness · {n}임계값 p &lt; {f["threshold"]:.2f}</div>
        </div>
        <div class="fa-legend">
          <span class="fa-legend-chip gender"><i></i>성별</span>
          <span class="fa-legend-chip age"><i></i>연령대</span>
        </div>
      </div>
      <div class="fa-bars">{"".join(groups)}</div>
      <div class="fa-note">{note}</div>
    </div>"""


def _mitigation_rows(m, targets):
    before, after = m["before"], m["after"]
    rows = []

    def add(label, b, a, change, change_cls, verdict, verdict_cls, bold=False, b_cls=""):
        rows.append((label, b, a, change, change_cls, verdict, verdict_cls, bold, b_cls))

    # DI Ratio: 1 에 가까워질수록 개선
    for attr, label in ATTRIBUTES:
        b, a = before["by_attribute"].get(attr), after["by_attribute"].get(attr)
        if not (b and a):
            continue
        better = abs(1 - a["di_ratio"]) <= abs(1 - b["di_ratio"])
        if not b["di_passed"] and a["di_passed"]:
            verdict, cls = "경고 해소", "ok"
        else:
            verdict, cls = ("개선", "ok") if better else ("악화", "ng")
        add(f"DI Ratio ({label})", f'{b["di_ratio"]:.2f}', f'{a["di_ratio"]:.2f}',
            f'{a["di_ratio"] - b["di_ratio"]:+.2f}', "ok" if better else "ng", verdict, cls,
            bold=True, b_cls="" if b["di_passed"] else "ng")

    # Equalized Odds: 차이가 작을수록 개선
    for key, label in [("equalized_odds_tpr_diff", "TPR 차이"), ("equalized_odds_fpr_diff", "FPR 차이")]:
        better = after[key] <= before[key]
        add(label, f"{before[key]:.2f}", f"{after[key]:.2f}", f"{after[key] - before[key]:+.2f}",
            "ok" if better else "ng", "개선" if better else "악화", "ok" if better else "ng")

    # 성능: 떨어져도 기준 이상이면 허용 (공정성과의 트레이드오프)
    for key, label, digits in [("auc", "AUC", 3), ("ks", "KS", 2)]:
        if before.get(key) is None or after.get(key) is None:
            continue
        diff, target = after[key] - before[key], targets.get(key, DEFAULT_TARGETS[key])
        if diff >= 0:
            verdict, cls = "유지", "ok"
        elif after[key] >= target:
            verdict, cls = f"허용 (≥ {target:.2f})", "warn"
        else:
            verdict, cls = f"미달 (≥ {target:.2f})", "ng"
        add(label, f"{before[key]:.{digits}f}", f"{after[key]:.{digits}f}", f"{diff:+.{digits}f}",
            "ok" if diff >= 0 else cls, verdict, cls, bold=key == "auc")

    if before.get("approval_rate") is not None and after.get("approval_rate") is not None:
        b, a = before["approval_rate"], after["approval_rate"]
        add("전체 승인율", f"{b:.1%}", f"{a:.1%}", f"{(a - b) * 100:+.1f}%p", "warn" if a < b else "ok", "—", "muted")
    return rows


def _mitigation_card(m, targets):
    rows = "".join(
        f'<tr><th class="{"bold" if bold else ""}">{label}</th>'
        f'<td class="{b_cls}">{b}</td><td>{a}</td>'
        f'<td class="{change_cls}">{change}</td>'
        f'<td class="fa-verdict {verdict_cls}">{verdict}</td></tr>'
        for label, b, a, change, change_cls, verdict, verdict_cls, bold, b_cls in _mitigation_rows(m, targets)
    )
    method = {
        "threshold_adjustment": "Threshold Adjustment (집단별 임계값 조정, 재학습 없음)",
        "reweighting": "Reweighting (샘플 가중치 재학습)",
    }.get(m.get("method"), escape(str(m.get("method", "—"))))
    return f"""
    <div class="fa-card">
      <div class="fa-card-head">
        <div>
          <div class="fa-title">Bias Mitigation 전 / 후 비교</div>
          <div class="fa-caption">GET /fairness/mitigation · 기법: {method}</div>
        </div>
        <span class="fa-chip amber">공정성 + 성능 동시 표기</span>
      </div>
      <table class="fa-table">
        <thead><tr><th>지표</th><th>완화 전</th><th>완화 후</th><th>변화</th><th>판정</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <div class="fa-note warn">⚠ 성능 하락은 실패가 아니라 트레이드오프의 기록. 단, 집단별 임계값 차등 적용은
        국내 규제상 논쟁적 — 리포트에서 별도 논의</div>
    </div>"""


def _protected_card(f):
    by_attr = f["by_attribute"]
    chips = []
    if "gender" in by_attr:
        groups = " / ".join(_ordered_groups("gender", by_attr["gender"]["approval_rates"]))
        chips.append(f"성별 {escape(groups)} — 시뮬레이터가 인위 부여 (GMSC에 성별 컬럼 없음)")
    if "age_group" in by_attr:
        groups = " / ".join(_group_label(g) for g in by_attr["age_group"]["approval_rates"])
        chips.append(f"연령대 {escape(groups)}")
    html = "".join(f'<span class="fa-chip">{c}</span>' for c in chips)
    return f"""
    <div class="fa-card fa-protected">
      <span class="fa-protected-label">보호속성</span>{html}
      <span class="fa-chip blue">예측 피처에서 제외 · 검증에만 사용</span>
    </div>"""


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------
st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")
render_page_header("공정성", "성별·연령대별 승인율과 DI Ratio · Equalized Odds 를 측정하고 Bias Mitigation 전후를 비교합니다")

try:
    fair = api_client.fairness()
    if fair is None:
        render_no_report("fairness")
except Exception as e:
    fair = None
    render_api_error("GET /fairness", e, "page")

if fair is not None:
    criteria = {**DEFAULT_CRITERIA, **fair.get("criteria", {})}
    _metric_cards(fair, criteria)

    try:
        targets = (api_client.metrics() or {}).get("targets", {})
    except Exception:
        targets = {}
    try:
        data = api_client.fairness_mitigation()
        mitigation = _mitigation_card(data, targets) if data else (
            '<div class="fa-card"><div class="fa-title">Bias Mitigation 전 / 후 비교</div>'
            f'{no_report_html("fairness_mitigation", "inline")}</div>')
    except Exception as e:
        mitigation = (f'<div class="fa-card"><div class="fa-title">Bias Mitigation 전 / 후 비교</div>'
                      f'{api_error_html("GET /fairness/mitigation", e, "inline")}</div>')
    st.html(f'<div class="fa-row2">{_approval_card(fair, criteria)}{mitigation}</div>')
    st.html(_protected_card(fair))
