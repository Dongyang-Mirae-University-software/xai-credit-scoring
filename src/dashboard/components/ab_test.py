"""기존 CB 모델 vs 대안신용평가 모델 (모델 성능 화면, 보너스 5, #140).

데이터: GET /ab-test (#150 결과). 같은 Test 고객을 두 모델로 각각 심사한 정책 비교 3가지와,
고객을 반씩 나눈 실제 실험(A/B)에서 그 차이가 보이는지를 함께 보여 준다.
대손율 = 승인 고객 중 실제 부도 비율 (금액 정보가 없어 건수 기준).
"""
from html import escape
from pathlib import Path

import streamlit as st

import api_client
from components import ui
from components.api_error import render_api_error
from components.empty_state import pending_html, render_no_report

CSS_PATH = Path(__file__).with_suffix(".css")
SECTION = "기존 CB 모델과 비교 (A/B 테스트 시뮬레이션)"
# (키, 제목, 설명, 비교해서 볼 지표)
POLICIES = [
    ("same_approval_rate", "같은 승인율이면", "대안 모델의 기준을 기존 모델과 승인율이 같게 맞췄어요", "bad_rate"),
    ("same_bad_rate", "같은 대손율이면", "대안 모델의 기준을 기존 모델과 대손율이 같게 맞췄어요", "approval_rate"),
    ("same_threshold", "같은 승인 기준이면", "두 모델 모두 부도 확률 {thr:.0%} 미만을 승인했어요", None),
]


def _pct(v, digits=1):
    return f"{v * 100:.{digits}f}%"


def _line(label, a, b, digits, better_low, focus):
    diff = round((b - a) * 100, digits)
    good = diff < 0 if better_low else diff > 0
    tone = "" if diff == 0 else ("ok" if good else "fail")
    change = f"{abs(diff):.{digits}f}%p" if diff == 0 else f"{diff:+.{digits}f}%p"   # 반올림해 0 이면 부호 없이
    return (f'<div class="ab-line{" focus" if focus else ""}"><span>{label}</span>'
            f'<em>{_pct(a, digits)}</em>{ui.icon("arrow", "#5B6577", 14)}<b>{_pct(b, digits)}</b>'
            f'<small class="{tone}">{change}</small></div>')


def _policy(key, title, desc, focus, data):
    pol = data["policies"]
    a, row = pol["a"], pol["policies"].get(key)
    if not row:
        return ""
    b = row["b"]
    lines = (_line("승인율", a["approval_rate"], b["approval_rate"], 1, False, focus == "approval_rate")
             + _line("대손율", a["bad_rate"], b["bad_rate"], 2, True, focus == "bad_rate"))
    thin = (row.get("segments") or {}).get("씬파일러")
    if thin:
        lines += _line("씬파일러 승인율", thin["a"]["approval_rate"], thin["b"]["approval_rate"], 1, False, False)
    if focus == "bad_rate":
        note = f'승인 고객 {b["n_approved"]:,}명 중 부도 {a["n_bad_approved"]:,}명 → {b["n_bad_approved"]:,}명'
    elif focus == "approval_rate":
        note = f'{b["n_approved"] - a["n_approved"]:+,}명을 더 승인해요'
    else:
        note = f'Test 고객 {a["n"]:,}명'
    return (f'<div class="ab-policy"><strong>{title}</strong>'
            f'<span class="ab-desc">{escape(desc.format(thr=pol.get("threshold_a", 0.5)))}</span>'
            f'{lines}<span class="ab-note">{note}</span></div>')


def _experiment(data):
    single, rep, size = data.get("ab_single") or {}, data.get("ab_repeated") or {}, data.get("required_sample_size") or {}
    test = single.get("bad_rate_test")
    if not test:
        return ""
    sig = test["p_value"] < 0.05
    text = (f'고객을 무작위로 반씩 나눠 한 번 실험하면 대손율 차이는 {test["diff"] * 100:+.2f}%p '
            f'(p = {test["p_value"]:.2f})로 {"우연이 아니에요" if sig else "우연과 구분되지 않아요"}.')
    if rep.get("n_rep"):
        text += f' {rep["n_rep"]:,}번 반복하면 {rep["share_significant_bad_rate"] * 100:.0f}%에서만 유의했어요.'
    if size.get("applicants_total"):
        text += (f' 이 차이를 검정력 {size["power"] * 100:.0f}%로 확인하려면 신청 고객이 '
                 f'모두 {size["applicants_total"]:,}명(한쪽 {size["applicants_per_arm"]:,}명) 필요해요.')
    return (f'<div class="ab-exp {"ok" if sig else "mid"}">{ui.icon("info", "#8A5200" if not sig else "#0B5E44", 18)}'
            f'<span><b>실제 A/B 실험이라면</b>{escape(text)}</span></div>')


def render():
    with st.container(key="bonus-ab", gap="small"):
        _render()


def _render():
    sub = ("같은 Test 고객을 두 모델로 각각 심사했어요. 대손율은 승인한 고객 중 실제 부도 비율이에요 "
           "(금액 정보가 없어 건수 기준).")
    st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>" + ui.section_head(SECTION, sub))
    try:
        data = api_client.ab_test()
    except Exception as e:
        render_api_error("GET /ab-test", e)
        return
    if data == api_client.NOT_READY:
        st.html(pending_html("GET /ab-test", "#139"))
        return
    if data is None:
        render_no_report("ab_test")
        return
    models, auc = data.get("models") or {}, data.get("test_auc") or {}
    who = "".join(
        f'<div><span>{tag}</span><b>{escape(name)}</b><small>{escape(models.get(k, "").split(": ", 1)[-1])}'
        f'{f" · AUC {auc[k]:.3f}" if auc.get(k) is not None else ""}</small></div>'
        for k, tag, name in (("a", "A", "기존 CB 모델"), ("b", "B", "대안신용평가 모델")))
    cards = "".join(_policy(*p, data) for p in POLICIES)
    st.html(f'<section class="ab-card"><div class="ab-models">{who}</div>'
            f'<div class="ab-policies">{cards}</div>{_experiment(data)}</section>')
