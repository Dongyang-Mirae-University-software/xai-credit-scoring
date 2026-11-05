"""공정성 페이지 (디자인 06)

결론(4개 항목 통과 여부) → 근거(집단별 승인율, 보정 전/후 전환) → 상세(보정 전후 비교 표).
데이터: GET /fairness, GET /fairness/mitigation (보정 전/후), GET /metrics (AUC 목표치).

승인율 비율(DI)은 디자인 정의대로 '가장 낮은 집단 승인율 ÷ 가장 높은 집단 승인율' 을 화면에서 계산하고
criteria.di_min(0.80) 이상이면 통과로 본다.
"""
from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st

import api_client
from components import ui
from components.api_error import render_api_error
from components.empty_state import render_no_report
from components.page_header import render_page_header

CSS_PATH = Path(__file__).with_suffix(".css")
DATASET_LABEL = "테스트셋"  # 백엔드 main_step4.py 는 Test 데이터로 공정성 지표를 계산한다
DEFAULT_CRITERIA = {"di_min": 0.8, "di_max": 1.25, "eo_max": 0.1}
ATTRIBUTES = [("gender", "성별"), ("age_group", "연령대")]
GENDER_LABELS = {"male": "남성", "female": "여성"}
DI_DOMAIN = (0.5, 1.3)   # 보정 전후 비교 표의 가로 범위
EO_DOMAIN = (0.0, 0.2)


def _group_label(group):
    if group in GENDER_LABELS:
        return GENDER_LABELS[group]
    if group.endswith("+"):
        return f"{group[:-1]}세 이상"
    return group.replace("-", "–") + "세"


def _ordered(attr, rates):
    order = ["male", "female"] if attr == "gender" else list(rates)
    return [g for g in order if g in rates] + [g for g in rates if g not in order]


def _di(info):
    rates = list((info or {}).get("approval_rates", {}).values())
    return min(rates) / max(rates) if rates and max(rates) > 0 else None


def _metrics_of(block, criteria):
    """by_attribute 묶음 → 4개 점검 항목 [(키, 이름, 기술 용어, 값, 기준 문구, 통과)]."""
    attrs = block.get("by_attribute", {})
    gender_di, age_di = _di(attrs.get("gender")), _di(attrs.get("age_group"))
    tpr = block.get("equalized_odds_tpr_diff")
    fpr = block.get("equalized_odds_fpr_diff")
    di_min, eo_max = criteria["di_min"], criteria["eo_max"]
    ok = lambda v, good: None if v is None else good(v)  # noqa: E731
    return [
        ("gender_di", "성별 승인율 비율", "DI Ratio, 남성 대비 여성", gender_di, f"{di_min:.2f} 이상",
         ok(gender_di, lambda v: v >= di_min)),
        ("age_di", "연령대 승인율 비율", "DI Ratio, 가장 높은 연령대 대비", age_di, f"{di_min:.2f} 이상",
         ok(age_di, lambda v: v >= di_min)),
        ("tpr", "부도 고객 적발률 차이", "Equalized Odds, TPR 차이", tpr, f"{eo_max:.2f} 이하",
         ok(tpr, lambda v: v <= eo_max)),
        ("fpr", "정상 고객 오판율 차이", "Equalized Odds, FPR 차이", fpr, f"{eo_max:.2f} 이하",
         ok(fpr, lambda v: v <= eo_max)),
    ]


def _v(value):
    return "—" if value is None else f"{value:.2f}"


def _pass_pill(ok):
    return ui.pill("통과" if ok else "미달", "ok" if ok else "fail") if ok is not None else ui.pill("—", "gray")


# ---------------------------------------------------------------------------
# 결론: 4개 항목 요약
# ---------------------------------------------------------------------------
def _josa_list(names):
    if not names:
        return ""
    text = "과 ".join(names) if len(names) == 2 else ", ".join(names)
    last = names[-1][-1]
    batchim = "가" <= last <= "힣" and (ord(last) - 0xAC00) % 28
    return text + ("이" if batchim else "가")


def _summary(now, before, mitigated):
    passed = [m for m in now if m[5]]
    all_ok = len(passed) == len(now)
    if all_ok:
        title = f"공정성 점검 {len(now)}개 항목을 모두 통과했어요"
        failed_before = [m[1] for m in before if m[5] is False] if mitigated else []
        sub = (f"보정 전에는 {_josa_list(failed_before)} 기준에 못 미쳤지만, 집단별 승인 기준선을 조정한 뒤 통과했어요."
               if failed_before else "보정 전후 모두 기준을 충족했어요.")
    else:
        failed = [m[1] for m in now if m[5] is False]
        title = f"공정성 점검 {len(now)}개 중 {len(failed)}개 항목이 미달이에요"
        sub = f"{_josa_list(failed)} 기준을 벗어났어요." + ("" if mitigated else " 보정 결과가 아직 없어요.")
    cells = "".join(
        f'<div class="fa-metric"><span>{name}</span>'
        f'<div><strong>{_v(value)}</strong>{_pass_pill(ok)}</div><small>기준 {rule}</small></div>'
        for _, name, _, value, rule, ok in now)
    tone = "ok" if all_ok else "fail"
    return f"""
    <section class="fa-summary">
      <div class="fa-banner {tone}">
        <span class="fa-mark">{ui.icon("check" if all_ok else "x", "#FFFFFF", 22)}</span>
        <div><strong>{title}</strong><span>{sub}</span></div>
      </div>
      <div class="fa-metrics">{cells}</div>
    </section>"""


# ---------------------------------------------------------------------------
# 근거: 집단별 승인율 막대
# ---------------------------------------------------------------------------
def _baseline(rates, criteria):
    """기준선 = 가장 높은 집단 승인율 × DI 최소 기준 (0.8). 승인율과 같은 0~1."""
    return max(rates.values()) * criteria["di_min"] if rates else 0


def _below(attrs, criteria):
    """기준선보다 승인율이 낮은 집단: [(속성 이름, [집단 이름…])]."""
    found = []
    for attr, label in ATTRIBUTES:
        rates = (attrs.get(attr) or {}).get("approval_rates", {})
        line = _baseline(rates, criteria)
        low = [_group_label(g) for g in _ordered(attr, rates) if rates[g] < line]
        if low:
            found.append((label, low))
    return found


def _state_strip(view, attrs, criteria, mitigated):
    """패널 위 한 줄: 기준선 아래 집단이 있으면 빨강(어느 집단·어느 항목이 미달), 없으면 초록."""
    below = _below(attrs, criteria)
    if below:
        groups = ", ".join(g for _, low in below for g in low)
        items = ", ".join(label for label, _ in below)
        return (f'<p class="fa-state fail">{view}: {escape(groups)} 승인율이 기준선보다 낮아 {escape(items)} 항목이 미달이에요. '
                '빨간 막대가 기준선 아래 집단이에요.</p>')
    how = "집단별 승인 기준선을 조정해 " if view == "보정 후" and mitigated else ""
    return f'<p class="fa-state ok">{view}: {how}모든 집단이 기준선을 넘었어요.</p>'


def _group_card(attr, label, info, criteria):
    rates = info.get("approval_rates", {})
    groups = _ordered(attr, rates)
    di = _di(info)
    line = _baseline(rates, criteria)
    rows = "".join(
        f'<div class="fa-bar-row{" low" if rates[g] < line else ""}"><span>{escape(_group_label(g))}</span>'
        f'<span class="fa-bar"><i style="width:{rates[g] * 100:.1f}%"></i></span>'
        f'<strong>{rates[g] * 100:.1f}%</strong></div>' for g in groups)
    ok = None if di is None else di >= criteria["di_min"]
    return f"""
    <div class="fa-card">
      <div class="fa-card-head"><h3>{label}</h3>
        <span>승인율 비율<strong class="{"fail" if ok is False else ""}">{_v(di)}</strong>{_pass_pill(ok)}</span></div>
      <div class="fa-bars" role="img" aria-label="{label}별 승인율">
        {rows}<i class="fa-line" style="left:calc(86px + (100% - 156px) * {line:.4f})"></i>
      </div>
      <div class="fa-line-note"><i></i>기준선: 가장 높은 집단 승인율의 {criteria["di_min"]:.0%} ({line * 100:.1f}%)</div>
    </div>"""


# ---------------------------------------------------------------------------
# 상세: 보정 전후 비교 표
# ---------------------------------------------------------------------------
def _pos(v, domain):
    lo, hi = domain
    return min(max((v - lo) / (hi - lo), 0), 1) * 100


def _range_cell(key, before, after, criteria):
    if key.endswith("_di"):
        domain, band = DI_DOMAIN, (criteria["di_min"], criteria.get("di_max", DI_DOMAIN[1]))
    else:
        domain, band = EO_DOMAIN, (0.0, criteria["eo_max"])
    b0, b1 = _pos(band[0], domain), _pos(band[1], domain)
    parts = [f'<i class="track"></i><i class="band" style="left:{b0:.1f}%;width:{b1 - b0:.1f}%"></i>']
    if before is not None and after is not None:
        p, q = _pos(before, domain), _pos(after, domain)
        parts.append(f'<i class="link" style="left:{min(p, q):.1f}%;width:{abs(q - p):.1f}%"></i>')
    if before is not None:
        parts.append(f'<i class="before" style="left:{_pos(before, domain):.1f}%"></i>')
    if after is not None:
        parts.append(f'<i class="after" style="left:{_pos(after, domain):.1f}%"></i>')
    return (f'<span class="fa-range"><span class="fa-range-track">{"".join(parts)}</span>'
            f'<span class="fa-range-axis"><span>{domain[0]:.1f}</span><span>{domain[1]:.1f}</span></span></span>')


def _compare_table(before, after, criteria):
    rows = []
    for (key, name, tech, b, rule, b_ok), (_, _, _, a, _, a_ok) in zip(before, after):
        fail = '<em>미달</em>' if b_ok is False else ""
        rows.append(f"""
        <div class="fa-row">
          <span class="fa-name"><strong>{name}</strong><small>{tech}</small></span>
          <span class="fa-rule">{rule}</span>
          {_range_cell(key, b, a, criteria)}
          <span class="num fa-before">{_v(b)}{fail}</span>
          <strong class="num">{_v(a)}</strong>
          <span class="num">{_pass_pill(a_ok)}</span>
        </div>""")
    legend = ('<span class="fa-legend"><span><i class="before"></i>보정 전</span><span><i class="after"></i>보정 후</span>'
              '<span><i class="band"></i>기준 범위</span></span>')
    return f"""
    <div class="fa-table" role="table" aria-label="보정 전후 공정성 지표">
      <div class="fa-row fa-headrow"><span>점검 항목</span><span>기준</span>{legend}
        <span class="num">보정 전</span><span class="num">보정 후</span><span class="num">결과</span></div>
      {"".join(rows)}
    </div>"""


def _change_card(mitigation, auc_target):
    b, a = mitigation.get("before", {}), mitigation.get("after", {})
    cells = []
    if b.get("approval_rate") is not None and a.get("approval_rate") is not None:
        cells.append(("전체 승인율", f'{b["approval_rate"] * 100:.1f}%', f'{a["approval_rate"] * 100:.1f}%'))
    note = ""
    if b.get("auc") is not None and a.get("auc") is not None:
        cells.append(("모델 정확도 (AUC)", f'{b["auc"]:.3f}', f'{a["auc"]:.3f}'))
        diff = a["auc"] - b["auc"]
        if auc_target is not None and a["auc"] < auc_target:
            note = f"정확도가 목표치({auc_target:.2f})보다 낮아졌어요."
        elif round(diff, 3) < 0:
            note = f"정확도는 소폭 낮아졌지만 목표치({auc_target:.2f})보다 높아요." if auc_target else "정확도는 소폭 낮아졌어요."
        else:
            note = "정확도는 그대로 유지됐어요." + (f" (목표치 {auc_target:.2f} 이상)" if auc_target else "")
    inner = "".join(f'<div><span>{label}</span><p><em>{x}</em> → <strong>{y}</strong></p></div>' for label, x, y in cells)
    return f'<div class="fa-change"><b>보정으로 달라진 점</b>{inner}<span class="fa-change-note">{note}</span></div>'


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------
st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")


# 기준선 조정 방식의 한계 (시안 하단 안내 박스). 보정 방식이 threshold_adjustment 일 때 보여 준다
LIMIT_NOTE = (
    '<div class="fa-limit"><div><strong>보정 방식의 한계</strong>'
    '<span>기준선 조정(Threshold Adjustment)은 심사 시점에 성별과 연령대를 사용해요. 실제 금융에서는 이것 자체가 '
    '차별 소지가 될 수 있어서, 학습 단계에서 보정하는 Reweighting이 더 적합해요. '
    '성별은 원본 데이터에 없어 시뮬레이터로 생성한 변수예요.</span></div></div>')


def _main():
    try:
        fairness = api_client.fairness()
        mitigation = api_client.fairness_mitigation()
    except Exception as e:
        fairness = mitigation = None
        render_page_header("공정성", "성별이나 나이 때문에 심사 결과가 불리하게 나오지 않는지 점검해요.")
        render_api_error("GET /fairness", e, "page")
        return

    if fairness is None and mitigation is None:
        render_page_header("공정성", "성별이나 나이 때문에 심사 결과가 불리하게 나오지 않는지 점검해요.")
        render_no_report("fairness")
        return

    criteria = {**DEFAULT_CRITERIA, **((fairness or {}).get("criteria") or {})}
    before_block = (mitigation or {}).get("before") or fairness
    after_block = (mitigation or {}).get("after")
    before = _metrics_of(before_block, criteria)
    after = _metrics_of(after_block, criteria) if after_block else before

    evaluated = (fairness or {}).get("evaluated_at") or (mitigation or {}).get("evaluated_at")
    try:
        evaluated = datetime.fromisoformat(evaluated).astimezone().strftime("%Y.%m.%d") if evaluated else "—"
    except ValueError:
        pass
    n = (fairness or {}).get("n")
    render_page_header("공정성", "성별이나 나이 때문에 심사 결과가 불리하게 나오지 않는지 점검해요.",
                       meta=[("점검 데이터", f"{DATASET_LABEL} {n:,}명" if n else "—"), ("마지막 점검", evaluated)])

    st.html(_summary(after, before, after_block is not None))

    # 집단별 승인율 (보정 전 / 후 전환)
    with st.container(key="fa-groups-head", horizontal=True, vertical_alignment="bottom"):
        st.html(ui.section_head("집단별 승인율", "같은 조건이라면 성별이나 나이와 상관없이 비슷한 비율로 승인되어야 해요."
                                + (" 버튼으로 보정 전후를 비교해 보세요." if after_block else "")))
        if after_block:
            view = st.radio("보정 상태", ["보정 전", "보정 후"], index=1, horizontal=True, key="fa_view",
                            label_visibility="collapsed")
        else:
            view = "보정 전"
    block = after_block if view == "보정 후" and after_block else before_block
    attrs = block.get("by_attribute", {})
    st.html('<div class="fa-panel">' + _state_strip(view, attrs, criteria, after_block is not None)
            + '<div class="fa-groups">' + "".join(
                _group_card(attr, label, attrs[attr], criteria) for attr, label in ATTRIBUTES if attr in attrs)
            + "</div></div>")

    # 보정 전후 비교
    if after_block:
        method = mitigation.get("method", "")
        method_text = "집단별 승인 기준선을 조정하는 방식(Threshold Adjustment)으로 편향을 줄였어요." if method == "threshold_adjustment" \
            else f"{escape(method)} 방식으로 편향을 줄였어요."
        try:
            auc_target = ((api_client.metrics() or {}).get("targets") or {}).get("auc")
        except Exception:
            auc_target = None
        st.html(f'<section class="fa-compare">{ui.section_head("보정 전후 비교", method_text)}'
                f'{_compare_table(before, after, criteria)}{_change_card(mitigation, auc_target)}'
                f'{LIMIT_NOTE if method == "threshold_adjustment" else ""}</section>')
    else:
        render_no_report("fairness_mitigation")


_main()
