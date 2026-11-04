"""공통 UI 조각: 아이콘, 상태 알약(pill), 정보 줄(라벨 + 굵은 값), 섹션 제목.

st.html 은 인라인 <svg> 를 지우므로 아이콘은 data URI 이미지(<img>)나 CSS mask 로 그린다.
이미지 안에서는 currentColor 가 통하지 않아 색을 직접 넘긴다.
"""
from html import escape
from urllib.parse import quote

# 디자인 시안의 아이콘 (viewBox 0 0 24 24, 선 아이콘). (내용, 선 두께)
ICONS = {
    "search": ('<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/>', 2),
    "account": ('<circle cx="12" cy="8" r="4"/><path d="M4 20c1.5-4 4.5-6 8-6s6.5 2 8 6"/>', 1.8),
    "screening": ('<circle cx="10" cy="8" r="4"/><path d="M3 20c1.2-3.6 3.8-5.5 7-5.5 1.3 0 2.5.3 3.5.9M15 18l2 2 4-4.5"/>', 1.8),
    "xai": ('<path d="M4 6h10M4 12h16M4 18h7"/>', 1.8),
    "fairness": ('<path d="M12 4v16M7 20h10M5 7h14M5 7l-3 6c0 1.7 1.3 3 3 3s3-1.3 3-3L5 7zM19 7l-3 6c0 1.7 1.3 3 3 3s3-1.3 3-3l-3-6z"/>', 1.6),
    "performance": ('<path d="M4 4v16h16"/><path d="M7 15c3-1 5-4 7-7 1.5-2 3-3 5-3"/>', 1.8),
    "stats": ('<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 10h16M10 10v10"/>', 1.8),
    "doc": ('<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M9 8h6M9 12h6M9 16h3"/>', 1.8),
    "plus": ('<path d="M12 5v14M5 12h14"/>', 2.2),
    "close": ('<path d="M6 6l12 12M18 6L6 18"/>', 2),
    "check": ('<path d="M5 12.5l4.5 4.5L19 7.5"/>', 2.4),
    "x": ('<path d="M7 7l10 10M17 7L7 17"/>', 2.6),
    "pencil": ('<path d="M4 20h4L19 9l-4-4L4 16v4z"/>', 1.8),
    "chevron": ('<path d="M9 6l6 6-6 6"/>', 2),
    "copy": ('<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>', 1.8),
    "download": ('<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>', 1.8),
    "info": ('<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>', 1.8),
    "warn": ('<path d="M12 3l10 18H2L12 3z"/><path d="M12 10v5M12 18v.5"/>', 1.8),
}

LOGO_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 28 28">'
            '<rect x="2" y="2" width="24" height="24" rx="6" fill="none" stroke="#FFFFFF" stroke-width="2"/>'
            '<path d="M8 18l4-5 3 3 5-7" fill="none" stroke="#7EA6FF" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def svg(name, color="#000", size=20):
    body, width = ICONS[name]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round">'
            f'{body}</svg>')


def data_uri(svg_text):
    return "data:image/svg+xml;utf8," + quote(svg_text)


def icon(name, color="#3A4456", size=20, cls="ui-icon"):
    """<img> 아이콘 (st.html 에서 살아남는다)."""
    return f'<img class="{cls}" src="{data_uri(svg(name, color, size))}" width="{size}" height="{size}" alt="">'


def logo():
    return f'<img src="{data_uri(LOGO_SVG)}" width="30" height="30" alt="">'


def mask_css(selector, name, size=20):
    """CSS mask 아이콘: 글자색(currentColor)을 그대로 따라간다. 링크·버튼의 ::before 에 쓴다."""
    uri = data_uri(svg(name, "#000", size))
    return (f'{selector}{{content:"";flex:0 0 {size}px;width:{size}px;height:{size}px;'
            f'background:currentColor;-webkit-mask:url("{uri}") center/contain no-repeat;'
            f'mask:url("{uri}") center/contain no-repeat;}}')


# ---------------------------------------------------------------------------
# 글/표 조각
# ---------------------------------------------------------------------------
PILL_KINDS = {"ok", "fail", "mid", "info", "gray"}


def pill(text, kind="gray"):
    """의미 있는 색만: ok(승인·통과) / fail(거절·미달) / mid(중간 위험) / info / gray."""
    kind = kind if kind in PILL_KINDS else "gray"
    return f'<span class="ui-pill {kind}">{escape(str(text))}</span>'


def info_strip(cells, cls=""):
    """'작은 라벨 + 굵은 값' 칸을 얇은 세로선으로 나눈 줄. cells: [(라벨, 값 HTML)]. 값은 이미 escape 된 HTML."""
    inner = "".join(f'<div class="ui-cell"><span>{escape(label)}</span><b>{value}</b></div>' for label, value in cells)
    return f'<div class="ui-strip {cls}">{inner}</div>'


def section_head(title, sub=None, right=""):
    sub_html = f'<p>{sub}</p>' if sub else ""
    return (f'<div class="ui-sec-head"><div><h2>{escape(title)}</h2>{sub_html}</div>'
            f'{right}</div>')


def term(main, tech):
    """은행원 용어 + 작은 기술 용어. 예: term('부도 고객 적발률 차이', 'Equalized Odds, TPR 차이')"""
    return f'{escape(main)}<small class="ui-term">{escape(tech)}</small>'
