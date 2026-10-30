"""통계 검증 페이지"""
from html import escape
from pathlib import Path

import streamlit as st

import api_client
from components.api_error import render_api_error
from components.empty_state import render_no_report
from components.page_header import render_page_header

CSS_PATH = Path(__file__).with_suffix(".css")

# /anova 응답의 그룹 이름 → 화면 표시 이름
GROUP_LABELS = {
    "전통모델": "전통 only",
    "대안모델": "대안 only",
    "통합모델": "통합 (전통+대안)",
    "LogisticRegression": "Logistic Regression",
}
COMPARISON_LABELS = {"LogisticRegression": "LR"}  # Tukey 비교 쌍은 칸이 좁아 줄여 쓴다
DATA_TYPE_LABELS = {"financial_only": "전통 피처", "alternative_only": "대안 피처", "integrated": "통합 피처"}
ETA_LEVELS = [("small", "0.01 – 0.06"), ("medium", "0.06 – 0.14"), ("large", "≥ 0.14")]

TESTS = [
    {
        "key": "test_1",
        "title": "검증 1 · 피처 그룹별 One-way ANOVA",
        "h0": "전통 / 대안 / 통합 피처 그룹의 평균 AUC가 같다",
        "group_col": "피처 그룹",
    },
    {
        "key": "test_2",
        "title": "검증 2 · 알고리즘별 One-way ANOVA",
        "h0": "LR / XGBoost / LightGBM의 평균 AUC가 같다",
        "group_col": "알고리즘",
    },
]


def _p(p):
    return "&lt; 0.001" if p < 0.001 else f"{p:.3f}"


def _eta_level(t):
    level = t.get("effect_size_interpretation")
    if level:
        return level
    eta = t["effect_size_eta_squared"]
    return "large" if eta >= 0.14 else "medium" if eta >= 0.06 else "small"


def _num(v, digits=3):
    return "—" if v is None else f"{v:.{digits}f}"


def _ci(ci, digits=3, sign=""):
    return f"[{ci[0]:{sign}.{digits}f}, {ci[1]:{sign}.{digits}f}]" if ci else "—"


def _comparison_label(text):
    for name, short in COMPARISON_LABELS.items():
        text = text.replace(name, short)
    return text


# ---------------------------------------------------------------------------
# 화면 조각
# ---------------------------------------------------------------------------
def _anova_card(spec, t, data):
    alpha = t.get("alpha", 0.05)
    sizes = t.get("group_sizes") or {}
    n_values = set(sizes.values())
    caption = f"H₀: {spec['h0']}"
    if spec["key"] == "test_2":
        data_type = (data.get("source") or {}).get("test_2_data_type")
        if data_type:
            caption += f" · {DATA_TYPE_LABELS.get(data_type, data_type)}"
    if len(n_values) == 1:
        caption += f" · <span class=\"st-nowrap\">그룹당 n = {n_values.pop()}</span>"

    texts = (data.get("interpretation") or {}).get(spec["key"]) or [
        "귀무가설 기각" if t["is_significant"] else "귀무가설 유지"]
    verdict = f'<span class="st-pill {"ok" if t["is_significant"] else "warn"}">{escape(texts[0])}</span>'

    level = _eta_level(t)
    level_cls = {"large": "ok", "medium": "warn"}.get(level, "")
    stats = f"""
      <div class="st-stats">
        <div class="st-stat"><span>F 통계량</span><b>{t["f_statistic"]:.1f}</b></div>
        <div class="st-stat"><span>p-value</span><b class="{"ok" if t["p_value"] < alpha else "warn"}">{_p(t["p_value"])}</b></div>
        <div class="st-stat"><span>η² (효과 크기)</span><b class="{level_cls}">{t["effect_size_eta_squared"]:.2f} · {level}</b></div>
      </div>"""

    means = t["group_means"]
    best = max(means, key=means.get) if means else None
    rows = "".join(
        f'<tr class="{"st-best" if g == best else ""}">'
        f'<th>{escape(GROUP_LABELS.get(g, g))}</th>'
        f'<td>{sizes.get(g, "—")}</td>'
        f'<td class="st-mean">{means[g]:.3f}</td>'
        f'<td>{_num((t.get("group_std") or {}).get(g))}</td>'
        f'<td>{_ci((t.get("group_ci95") or {}).get(g))}</td></tr>'
        for g in t["groups"] if g in means
    )
    return f"""
    <div class="st-card">
      <div class="st-card-head">
        <div>
          <div class="st-title">{spec["title"]}</div>
          <div class="st-caption">{caption}</div>
        </div>
        {verdict}
      </div>
      {stats}
      <table class="st-table">
        <thead><tr><th>{spec["group_col"]}</th><th>n</th><th>평균 AUC</th><th>표준편차</th><th>95% CI</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


def _tukey_card(spec, t):
    alpha = t.get("alpha", 0.05)
    title = f"Tukey HSD · {spec['title'].split(' · ')[0]} 사후검정"
    tukey = t.get("post_hoc_tukey") or []
    if not t["is_significant"] or not tukey:
        return f"""
        <div class="st-card">
          <div class="st-title">{title}</div>
          <div class="st-caption">p ≥ {alpha:.2f} 이므로 사후검정을 하지 않았습니다</div>
        </div>"""

    same = [r["comparison"] for r in tukey if not r["significant"]]
    if spec["key"] == "test_2" and same:
        a, b = (_comparison_label(x) for x in same[0].split(" vs "))
        caption = f"{escape(a)}와 {escape(b)}은 통계적으로 구분되지 않음 → 선택 근거는 응답시간·SHAP 호환성"
    else:
        caption = f"p &lt; {alpha:.2f} 이므로 필수 · 어느 그룹 쌍이 다른지 확인"

    rows = "".join(
        f'<tr><th>{escape(_comparison_label(r["comparison"]))}</th>'
        f'<td>{r["mean_diff"]:+.3f}</td>'
        f'<td class="{"" if r["significant"] else "warn"}">{_p(r["p_adj"])}</td>'
        f'<td>{_ci(r.get("ci95"), sign="+")}</td>'
        f'<td class="st-reject {"ok" if r["significant"] else "warn"}">{"✓ 기각" if r["significant"] else "✗ 유지"}</td></tr>'
        for r in tukey
    )
    return f"""
    <div class="st-card">
      <div class="st-title">{title}</div>
      <div class="st-caption">{caption}</div>
      <table class="st-table">
        <thead><tr><th>비교 쌍</th><th>평균 차이</th><th>p-adj</th><th>95% CI</th><th>기각</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


def _footer_cards(data):
    chips = "".join(f'<span class="st-chip {level}">{rng}&nbsp; {level}</span>' for level, rng in ETA_LEVELS)
    source = data.get("source") or {}
    detail = []
    if source.get("metric"):
        detail.append(f'MLflow {escape(source["metric"])} 메트릭 히스토리에서 계산')
    if source.get("runs_used"):
        detail.append(f'<span class="st-nowrap">런 {source["runs_used"]}개</span>')
    return f"""
    <div class="st-row3">
      <div class="st-card">
        <div class="st-subtitle">η² 해석 기준</div>
        <div class="st-chips st-eta">{chips}</div>
      </div>
      <div class="st-card">
        <div class="st-subtitle warn">⚠ 독립성 가정의 한계 (리포트 명시)</div>
        <div class="st-caption">폴드 AUC는 같은 데이터를 재분할해 얻은 값이라 완전히 독립이 아니다.
          시드를 바꿔 반복해도 p-value 는 실제보다 작게 나올 수 있다.</div>
      </div>
      <div class="st-card">
        <div class="st-subtitle">데이터 출처</div>
        <div class="st-chips"><span class="st-chip blue">GET /anova → {{test_1, test_2}}</span></div>
        <div class="st-caption">{" · ".join(detail) or "MLflow 폴드별 AUC"}</div>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------
st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")
render_page_header("통계 검증", "ANOVA 2종과 Tukey HSD 로 대안 데이터의 효용과 알고리즘 선택을 통계적으로 정당화합니다")

try:
    anova = api_client.anova()
    if anova is None:
        render_no_report("anova")
except Exception as e:
    anova = None
    render_api_error("GET /anova", e, "page")

if anova is not None:
    st.html('<div class="st-row2 st-anova">'
            + "".join(_anova_card(spec, anova[spec["key"]], anova) for spec in TESTS) + "</div>")
    st.html('<div class="st-row2 st-tukey">'
            + "".join(_tukey_card(spec, anova[spec["key"]]) for spec in TESTS) + "</div>")
    st.html(_footer_cards(anova))
