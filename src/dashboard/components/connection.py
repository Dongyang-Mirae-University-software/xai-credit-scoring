"""API 연결 상태 (디자인 상태 화면 #loading · #api-down). 모든 페이지 공통 (app.py 에서 부른다).

- 첫 진입과 '다시 연결' 때: GET /health 를 기다리는 동안 본문은 스켈레톤만, 사이드바는 '연결 중'
- 연결 실패: 본문 맨 위 빨간 경고 띠 + '다시 연결', 사이드바 '연결 안 됨'. 화면은 is_down() 으로 심사 버튼을 끈다
- 그 뒤에는 HEALTH_CACHE_TTL 초마다 다시 확인한다 (서버가 꺼지면 다음 확인 때 실패 상태로)
"""
from pathlib import Path

import requests
import streamlit as st

import api_client
import config
from components import sidebar

CSS_PATH = Path(__file__).with_suffix(".css")


def _health_now():
    try:
        return api_client.request("GET", "/health", timeout=config.HEALTH_TIMEOUT)
    except requests.RequestException:
        return None


@st.cache_data(ttl=config.HEALTH_CACHE_TTL, show_spinner=False)
def _health_cached(base_url):
    return _health_now()


def _bar(width, height=14):
    return f'<span class="cn-skel" style="width:{width};height:{height}px"></span>'


def _skeleton(page):
    """시안 #loading: 페이지 제목 + 회색 막대만 (안내 문구·카드 없음)."""
    head = f'<div class="ui-page-head"><div><h1>{page.title}</h1>{PAGE_SUBTITLES.get(page.title, "")}</div></div>'
    rows = "".join('<div class="cn-skel-row">' + "".join(_bar(w) for w in ("70%", "50%", "40%", "45%", "40%", "60%"))
                   + "</div>" for _ in range(5))
    return (f'<style>{CSS_PATH.read_text(encoding="utf-8")}</style><div class="cn-loading" aria-busy="true" '
            f'aria-label="불러오는 중">{head}'
            f'<div class="cn-skel-card">{_bar("140px")}{_bar("100%", 52)}</div>'
            f'<div class="cn-skel-list">{_bar("160px", 20)}<div class="cn-skel-table"><div class="cn-skel-th"></div>'
            f'{rows}</div></div></div>')


# 시안 #loading 은 제목 아래 설명까지 보인다 (고객 심사 화면 기준). 다른 화면도 같은 자리의 설명을 쓴다
PAGE_SUBTITLES = {
    "고객 심사": "<p>기존 고객은 ID로 불러오고, 처음 오신 고객은 신규 심사로 시작하세요.</p>",
    "XAI 분석": "<p>이 고객이 왜 이런 결과를 받았는지 설명해요.</p>",
}


def _reconnect():
    st.session_state.api_reconnect = True
    _health_cached.clear()


def check(page, status_slot):
    """연결을 확인해 사이드바 상태를 채우고, 실패하면 경고 띠를 그린다. /health 응답(실패면 None)을 돌려준다."""
    ss = st.session_state
    if "api_health" not in ss or ss.pop("api_reconnect", False):
        status_slot.html(sidebar.status_html("loading"))
        skeleton = st.empty()
        skeleton.html(_skeleton(page))
        health = _health_now()
        _health_cached.clear()
        skeleton.empty()
    else:
        health = _health_cached(config.API_BASE_URL)
    ss.api_health = health
    status_slot.html(sidebar.status_html("down" if health is None else "ok", health))
    if health is None:
        with st.container(key="cn-down", horizontal=True, vertical_alignment="center"):
            st.html(f'<style>{CSS_PATH.read_text(encoding="utf-8")}</style>'
                    '<div class="cn-down-text"><strong>API 서버에 연결할 수 없어요</strong>'
                    '<span>심사와 지표 조회가 잠시 중단돼요. 서버가 켜져 있는지 확인한 뒤 다시 연결해 주세요.</span></div>')
            st.button("다시 연결", key="cn-reconnect", on_click=_reconnect)
    return health


def is_down():
    return st.session_state.get("api_health", {}) is None
