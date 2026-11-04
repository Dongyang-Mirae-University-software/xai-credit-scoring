"""통계 검증 페이지 (디자인 08)

결론(두 ANOVA 요약 표) → 근거/상세(검증별 결론 문장 · 그룹별 평균 AUC 막대 · Tukey HSD 표).
데이터: GET /anova, 최종 모델 강조는 GET /health 의 model_version.
"""
import math
import re
from html import escape
from pathlib import Path

import streamlit as st

import api_client
import features as F
from components import ui
from components.api_error import render_api_error
from components.empty_state import render_no_report
from components.page_header import render_page_header

CSS_PATH = Path(__file__).with_suffix(".css")
SUBTITLE = "성능 차이가 우연이 아닌지 분산분석(ANOVA)으로 확인해요."
GROUP_LABELS = {
    "전통모델": "전통 데이터만", "대안모델": "대안 데이터만", "통합모델": "통합 (전통 + 대안)",
    "LogisticRegression": "Logistic Regression",
}
SHORT = {"LogisticRegression": "LR"}  # 비교 쌍 이름은 짧게
DATA_TYPE_GROUP = {"financial_only": "전통모델", "alternative_only": "대안모델", "integrated": "통합모델"}
ALGORITHM_GROUP = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "logistic": "LogisticRegression"}
TESTS = [
    ("test_1", "검증 1. 피처 구성별 비교", "대안 데이터를 넣으면 성능이 정말 좋아질까?"),
    ("test_2", "검증 2. 알고리즘별 비교", "어떤 알고리즘을 최종 모델로 쓸까?"),
]


def _p(p):
    return "&lt; 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def _label(group):
    return GROUP_LABELS.get(group, group)


def _short(text):
    for k, v in SHORT.items():
        text = text.replace(k, v)
    return text


def _signed(v):
    return f"{v:+.3f}".replace("-", "−")


def _eta_level(t):
    level = t.get("effect_size_interpretation")
    if level:
        return level.capitalize()
    eta = t["effect_size_eta_squared"]
    return "Large" if eta >= 0.14 else "Medium" if eta >= 0.06 else "Small"


def _final_groups():
    """최종 모델이 속한 그룹: (피처 구성 그룹, 알고리즘 그룹). 모델 이름 예: lightgbm_is_unbalance_integrated_v1.0"""
    try:
        name = (api_client.health().get("model_version") or "").lower()
    except Exception:
        return None, None
    data = next((g for k, g in DATA_TYPE_GROUP.items() if k in name), None)
    algo = next((g for k, g in ALGORITHM_GROUP.items() if name.startswith(k)), None)
    return data, algo


# ---------------------------------------------------------------------------
# 결론: 요약 표
# ---------------------------------------------------------------------------
def _summary(anova):
    tests = [(key, title, anova[key]) for key, title, _ in TESTS if key in anova]
    sig = [t for _, _, t in tests if t["is_significant"]]
    alpha = tests[0][2].get("alpha", 0.05) if tests else 0.05
    if len(sig) == len(tests):
        tone, title = "ok", ("두 검증 모두 통계적으로 유의한 차이가 있어요" if len(tests) == 2 else "통계적으로 유의한 차이가 있어요")
        sub = (f"p값이 {alpha:g}보다 작아 그룹 간 AUC 차이가 우연이 아니에요. "
               "어느 그룹끼리 다른지는 아래 사후 검정(Tukey HSD)에서 확인해요.")
    else:
        tone, title = "mid", f"{len(tests)}개 검증 중 {len(sig)}개만 유의한 차이가 있어요"
        sub = f"p값이 {alpha:g} 이상인 검증은 그룹 간 AUC 차이를 우연과 구분할 수 없어요."
    rows = "".join(f"""
      <div class="stt-row">
        <span class="stt-name"><strong>{title}</strong><small>One-way ANOVA, 그룹 {len(t["groups"])}개</small></span>
        <strong class="num">{t["f_statistic"]:.1f}</strong>
        <strong class="num">{_p(t["p_value"])}</strong>
        <strong class="num">{t["effect_size_eta_squared"]:.2f}</strong>
        <span class="num">{_eta_level(t)}</span>
        <span class="num">{ui.pill("유의함" if t["is_significant"] else "차이 없음", "ok" if t["is_significant"] else "gray")}</span>
      </div>""" for _, title, t in tests)
    return f"""
    <section class="stt-summary">
      <div class="stt-banner {tone}"><span class="stt-mark">{ui.icon("check" if tone == "ok" else "info", "#FFFFFF", 22)}</span>
        <div><strong>{title}</strong><span>{sub}</span></div></div>
      <div role="table" aria-label="ANOVA 검증 결과">
        <div class="stt-row stt-headrow"><span>검증</span><span class="num">F값</span><span class="num">p값</span>
          <span class="num">η² (효과 크기)</span><span class="num">효과 해석</span><span class="num">결과 (α = {alpha:g})</span></div>
        {rows}
      </div>
      <div class="stt-foot">η² 해석 기준: 0.01 이상 Small, 0.06 이상 Medium, 0.14 이상 Large</div>
    </section>"""


# ---------------------------------------------------------------------------
# 검증별 상세
# ---------------------------------------------------------------------------
def _subject(name, with_batchim="이", without="가"):
    """조사 붙이기: 괄호 설명은 빼고 받침을 본다. '통합 (전통 + 대안)' → '통합 (전통 + 대안)이'"""
    return name + F.josa(re.sub(r"\s*\(.*\)$", "", name), with_batchim, without)


def _conclusion(t, final):
    means = t["group_means"]
    if not t["is_significant"]:
        return "그룹 간 평균 AUC 차이가 통계적으로 유의하지 않아요."
    best = max(means, key=means.get)
    tukey = t.get("post_hoc_tukey") or []
    best_name = _label(best)
    key = best.replace("모델", "")  # 비교 쌍은 '전통 vs 통합' 처럼 '모델' 없이 온다
    involving = [r for r in tukey if key in r["comparison"].split(" vs ")]
    same = [r for r in involving if not r["significant"]]
    if involving and not same:
        text = f"{_subject(best_name)} 다른 그룹보다 통계적으로 유의하게 높아요."
    elif same:
        other = same[0]["comparison"]
        a, b = (_label(_short(x)) for x in other.split(" vs "))
        text = f"{best_name}의 평균이 가장 높지만, {escape(a)}{F.josa(a, '과', '와')} {escape(_subject(b, '은', '는'))} 차이가 없어요."
    else:
        text = f"{best_name}의 평균 AUC가 가장 높아요."
    if final and final in means and final != best:
        text += f" 최종 모델은 {_subject(_label(final), '을', '를')} 선택했어요."
    return text


def _bars(t, highlight):
    means = t["group_means"]
    values = [means[g] for g in t["groups"] if g in means]
    lo = math.floor((min(values) - 0.08) * 20) / 20
    hi = math.ceil((max(values) + 0.02) * 20) / 20
    rows = "".join(
        f'<div class="stt-bar-row"><span>{escape(_label(g))}</span>'
        f'<span class="stt-bar"><i class="{"on" if g == highlight else ""}" '
        f'style="width:{(means[g] - lo) / (hi - lo) * 100:.1f}%"></i></span>'
        f'<strong>{means[g]:.3f}</strong></div>' for g in t["groups"] if g in means)
    return (f'<div class="stt-col"><span class="stt-col-title">그룹별 평균 AUC</span>'
            f'<div class="stt-bars" role="img" aria-label="그룹별 평균 AUC 막대 차트">{rows}</div>'
            f'<span class="stt-note">막대는 AUC {lo:.2f}부터 그렸어요.</span></div>')


def _tukey(t):
    tukey = t.get("post_hoc_tukey") or []
    if not tukey:
        body = '<div class="stt-empty">유의한 차이가 없어 사후 검정을 하지 않았어요.</div>'
    else:
        body = "".join(
            f'<div class="stt-tk-row"><strong>{escape(_short(r["comparison"]))}</strong>'
            f'<span class="num">{_signed(r["mean_diff"])}</span><span class="num">{_p(r["p_adj"])}</span>'
            f'<span class="num">{ui.pill("유의함" if r["significant"] else "차이 없음", "ok" if r["significant"] else "gray")}</span></div>'
            for r in tukey)
    return (f'<div class="stt-col stt-tk"><span class="stt-col-title">사후 검정 (Tukey HSD)</span>'
            f'<div role="table" aria-label="Tukey HSD 결과"><div class="stt-tk-row stt-tk-head"><span>비교</span>'
            f'<span class="num">평균 차이</span><span class="num">조정 p값</span><span class="num">결과</span></div>{body}</div></div>')


def _test_section(key, title, question, t, highlight):
    return f"""
    <section class="stt-section">
      {ui.section_head(title, question)}
      <div class="stt-card">
        <p class="stt-conclusion"><strong>결론.</strong> {_conclusion(t, highlight)}</p>
        <div class="stt-grid">{_bars(t, highlight)}{_tukey(t)}</div>
      </div>
    </section>"""


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------
st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")


def _main():
    try:
        anova = api_client.anova()
    except Exception as e:
        render_page_header("통계 검증", SUBTITLE)
        render_api_error("GET /anova", e, "page")
        return
    if anova is None:
        render_page_header("통계 검증", SUBTITLE)
        render_no_report("anova")
        return

    cv = (anova.get("source") or {}).get("cv") or {}   # 요청해 둔 필드: {"n_splits": 5, "n_repeats": 5}
    cv_text = (f'{cv["n_splits"]}-Fold × {cv["n_repeats"]}회 반복 CV' if cv.get("n_splits") and cv.get("n_repeats")
               else "교차검증(CV) 폴드별 AUC")
    sizes = sorted({n for key, _, _ in TESTS for n in (anova.get(key, {}).get("group_sizes") or {}).values()})
    size_text = "AUC " + " / ".join(f"{n}개" for n in sizes) if sizes else "—"
    render_page_header("통계 검증", SUBTITLE, meta=[("검증 데이터", cv_text), ("그룹별 표본", size_text)])

    st.html(_summary(anova))
    data_group, algo_group = _final_groups()
    for (key, title, question), highlight in zip(TESTS, (data_group, algo_group)):
        if key in anova:
            st.html(_test_section(key, title, question, anova[key], highlight))


_main()
