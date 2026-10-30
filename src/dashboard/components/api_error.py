"""API 호출 실패 안내 카드: 예외 원문 대신 원인·해결 방법을 보여 주고, 원문은 '자세히'에 접어 둔다."""
from html import escape
from pathlib import Path
from urllib.parse import urlparse

import requests
import streamlit as st

import config

CSS_PATH = Path(__file__).with_suffix(".css")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}


def _server_command():
    """API 주소가 내 컴퓨터면 서버 실행 명령을 안내한다 (프로젝트 루트에서 실행)."""
    url = urlparse(config.API_BASE_URL)
    if url.hostname not in LOCAL_HOSTS:
        return None
    return f"uvicorn src.api.main:app --port {url.port or 8000}"


def _validation_messages(response):
    """FastAPI 422 응답의 detail 목록 → '필드: 메시지' 몇 줄."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        return []
    if not isinstance(detail, list):
        return [str(detail)] if detail else []
    return [f'{d.get("loc", ["?"])[-1]}: {d.get("msg", "")}' for d in detail[:3]]


def describe(exc):
    """예외 → (종류, 제목, 설명, 해결 안내 HTML)."""
    if isinstance(exc, requests.ConnectionError):
        command = _server_command()
        hint = (f'프로젝트 폴더에서 API 서버를 켠 뒤 새로고침해 주세요<code>{escape(command)}</code>' if command
                else f'API 주소 <code>{escape(config.API_BASE_URL)}</code> 가 맞는지 확인해 주세요')
        return "offline", "API 서버에 연결할 수 없어요", "FastAPI 서버가 꺼져 있거나 주소가 달라요.", hint
    if isinstance(exc, requests.Timeout):
        return ("slow", "API 응답이 너무 늦어요",
                f"{config.API_TIMEOUT}초 안에 응답이 오지 않았어요.", "잠시 후 새로고침해 주세요")
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        if status == 404:
            return "error", "API에 이 기능이 없어요", "서버가 404를 돌려줬어요.", "API 서버가 최신 코드인지 확인해 주세요"
        if status == 422:
            lines = "".join(f"<li>{escape(m)}</li>" for m in _validation_messages(exc.response))
            return ("warn", "입력값을 확인해 주세요", "API가 요청 값을 받아들이지 않았어요.",
                    f"<ul>{lines}</ul>" if lines else "입력 범위를 확인해 주세요")
        if status >= 500:
            return "error", "API 서버에서 오류가 났어요", f"서버가 {status}을 돌려줬어요.", "API 서버 터미널의 로그를 확인해 주세요"
        return "error", "API 요청이 거절됐어요", f"서버가 {status}을 돌려줬어요.", "잠시 후 다시 시도해 주세요"
    return ("error", "응답을 화면에 그리지 못했어요", "API 응답 형식이 대시보드가 기대한 형식과 달라요.",
            "API와 대시보드가 같은 버전인지 확인해 주세요")


def api_error_html(endpoint, exc, variant="banner"):
    """오류 안내 HTML.
    banner: 페이지 중간 한 줄 카드 / page: 페이지 내용 전체가 실패 → 남는 높이를 채우고 가운데 정렬 /
    inline: 다른 카드 안에 넣을 때 (테두리 없음, 가운데 정렬)"""
    kind, title, message, hint = describe(exc)
    icon = {"offline": "⚡", "slow": "⏱", "warn": "!"}.get(kind, "!")
    return f"""
    <style>{CSS_PATH.read_text(encoding='utf-8')}</style>
    <div class="ae-box {kind} {variant}">
      <span class="ae-icon">{icon}</span>
      <div class="ae-body">
        <div class="ae-head"><b>{title}</b><span class="ae-endpoint">{escape(endpoint)}</span></div>
        <div class="ae-message">{message}</div>
        <div class="ae-hint">{hint}</div>
        <details class="ae-detail"><summary>오류 원문 보기</summary><pre>{escape(f"{type(exc).__name__}: {exc}")}</pre></details>
      </div>
    </div>"""


def render_api_error(endpoint, exc, variant="banner"):
    st.html(api_error_html(endpoint, exc, variant))
