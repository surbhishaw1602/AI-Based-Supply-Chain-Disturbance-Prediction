"""
ui_theme.py - visual layer for the AI-Based Supply Chain Disturbance Prediction project.

Pure presentation helpers: nothing in here touches the database, GPS,
routing, weather or ML logic.
"""

from html import escape

import streamlit as st

# --------------------------------------------------
# PAGE META: icon, title and one-line description
# --------------------------------------------------
PAGE_META = {
    "Dashboard": ("📊", "Control Tower", "Fleet-wide overview of shipments, events and route risks."),
    "Register Shipment": ("📦", "Register Shipment", "Add a new shipment; coordinates are found automatically."),
    "GPS Tracking": ("📍", "Live GPS Tracking", "Real-time position, route progress and ETA of the active shipment."),
    "Weather Monitoring": ("🌦️", "Weather Monitoring", "Forecasts along the route and at both endpoints."),
    "Disturbance Monitoring": ("📢", "Disturbance Monitoring", "RSS news scan for strikes, blockades and road incidents."),
    "ML Dataset Collection": ("🗂️", "ML Dataset Collection", "Collect and verify observations used to train the model."),
    "ML Model Training": ("🤖", "ML Model Training", "Train and evaluate the Random Forest risk classifier."),
    "ML Prediction Dashboard": ("🔮", "Seven-Day Risk Prediction", "Forecast disturbance risk for every route checkpoint."),
}

NAV_ICONS = {name: meta[0] for name, meta in PAGE_META.items()}

# Status -> (emoji, css class)
STATUS_STYLE = {
    "In Transit": ("🚚", "st-blue"),
    "Delivered": ("✅", "st-green"),
    "Cancelled": ("⛔", "st-red"),
    "Returned": ("↩️", "st-amber"),
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  --acc:#38BDF8; --acc2:#6366F1; --ok:#22C55E; --warn:#F59E0B; --bad:#EF4444;
  --card:rgba(148,163,184,.08); --line:rgba(148,163,184,.22); --muted:rgba(148,163,184,.95);
}
html, body, [class*="css"], .stApp { font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif; }

/* ---------- layout ---------- */
.block-container{ padding-top:1.4rem; padding-bottom:3rem; max-width:1500px; }
header[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer{ visibility:hidden; }
hr{ margin:1.2rem 0 !important; border-color:var(--line) !important; }

/* ---------- hero ---------- */
.hero{
  position:relative; overflow:hidden; border-radius:20px; padding:26px 30px; margin-bottom:18px;
  background:linear-gradient(120deg,#0EA5E9 0%,#6366F1 55%,#8B5CF6 100%);
  box-shadow:0 12px 34px rgba(99,102,241,.28); color:#fff;
}
.hero:after{ content:""; position:absolute; right:-60px; top:-70px; width:260px; height:260px; border-radius:50%;
  background:radial-gradient(circle,rgba(255,255,255,.28),rgba(255,255,255,0) 70%); }
.hero h1{ margin:0; font-size:1.85rem; font-weight:800; letter-spacing:-.4px; color:#fff; padding:0; }
.hero p{ margin:6px 0 0; font-size:.98rem; opacity:.92; color:#fff; }
.hero .chips{ margin-top:14px; display:flex; flex-wrap:wrap; gap:8px; }
.chip{ background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.28); backdrop-filter:blur(6px);
  padding:4px 12px; border-radius:999px; font-size:.78rem; font-weight:600; color:#fff; }

/* ---------- KPI cards ---------- */
.kpi{
  border:1px solid var(--line); background:var(--card); border-radius:16px; padding:16px 18px; height:100%;
  position:relative; overflow:hidden; transition:transform .15s ease, box-shadow .15s ease;
}
.kpi:hover{ transform:translateY(-2px); box-shadow:0 8px 22px rgba(0,0,0,.18); }
.kpi:before{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--kc,var(--acc)); }
.kpi .k-top{ display:flex; justify-content:space-between; align-items:center; }
.kpi .k-label{ font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); font-weight:600; }
.kpi .k-icon{ font-size:1.25rem; opacity:.9; }
.kpi .k-value{ font-size:2rem; font-weight:800; line-height:1.15; margin-top:6px; }
.kpi .k-sub{ font-size:.78rem; color:var(--muted); margin-top:2px; }

/* ---------- section titles ---------- */
.sec{ display:flex; align-items:center; gap:10px; margin:.4rem 0 .8rem; }
.sec .bar{ width:5px; height:22px; border-radius:4px; background:linear-gradient(var(--acc),var(--acc2)); }
.sec .t{ font-size:1.15rem; font-weight:700; }
.sec .s{ font-size:.82rem; color:var(--muted); margin-left:4px; }

/* ---------- badges ---------- */
.badge{ display:inline-block; padding:3px 11px; border-radius:999px; font-size:.74rem; font-weight:700; border:1px solid transparent; }
.st-blue{ background:rgba(56,189,248,.14); color:#38BDF8; border-color:rgba(56,189,248,.35); }
.st-green{ background:rgba(34,197,94,.14); color:#22C55E; border-color:rgba(34,197,94,.35); }
.st-red{ background:rgba(239,68,68,.14); color:#EF4444; border-color:rgba(239,68,68,.35); }
.st-amber{ background:rgba(245,158,11,.14); color:#F59E0B; border-color:rgba(245,158,11,.35); }
.st-grey{ background:rgba(148,163,184,.14); color:#94A3B8; border-color:rgba(148,163,184,.35); }

/* ---------- shipment / event cards ---------- */
.card{ border:1px solid var(--line); background:var(--card); border-radius:14px; padding:14px 16px; margin-bottom:10px; }
.card .c-title{ font-weight:700; font-size:.98rem; }
.card .c-meta{ color:var(--muted); font-size:.82rem; margin-top:3px; }
.route-line{ display:flex; align-items:center; gap:8px; font-weight:600; flex-wrap:wrap; }
.route-line .arrow{ color:var(--acc); }

/* ---------- status board ---------- */
.board-head{ display:flex; justify-content:space-between; align-items:center; padding:10px 14px; border-radius:12px;
  border:1px solid var(--line); border-top:4px solid var(--bc,var(--acc)); background:var(--card); font-weight:700; margin-bottom:4px; }
.board-count{ background:var(--bc,var(--acc)); color:#fff; border-radius:999px; padding:1px 11px; font-size:.82rem; }
.board-sub{ font-size:.76rem; color:var(--muted); margin:0 0 10px 4px; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{ border-right:1px solid var(--line); }
section[data-testid="stSidebar"] .block-container{ padding-top:1rem; }
.brand{ display:flex; align-items:center; gap:12px; padding:6px 4px 14px; }
.brand .logo{ width:44px; height:44px; border-radius:12px; display:grid; place-items:center; font-size:1.5rem;
  background:linear-gradient(135deg,#0EA5E9,#6366F1); box-shadow:0 6px 16px rgba(99,102,241,.4); }
.brand .name{ font-weight:800; font-size:1.05rem; line-height:1.1; }
.brand .tag{ font-size:.72rem; color:var(--muted); }
.nav-label{ font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-weight:700; margin:14px 0 4px 4px; }

/* radio -> nav buttons */
section[data-testid="stSidebar"] div[role="radiogroup"]{ gap:4px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label{
  padding:9px 12px; border-radius:11px; border:1px solid transparent; transition:all .12s ease; width:100%;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover{ background:var(--card); border-color:var(--line); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked){
  background:linear-gradient(120deg,rgba(14,165,233,.25),rgba(99,102,241,.25)); border-color:rgba(99,102,241,.55);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child{ display:none; } /* hide radio dot */
section[data-testid="stSidebar"] div[role="radiogroup"] p{ font-weight:600; font-size:.93rem; }

.ship-card{ border:1px solid var(--line); background:var(--card); border-radius:14px; padding:12px 14px; margin-top:6px; }
.ship-card .sc-route{ font-weight:700; font-size:.9rem; }
.ship-card .sc-meta{ color:var(--muted); font-size:.76rem; margin-top:4px; }

/* ---------- native widgets ---------- */
div[data-testid="stMetric"]{
  border:1px solid var(--line); background:var(--card); border-radius:14px; padding:12px 16px;
}
div[data-testid="stMetricLabel"] p{ font-size:.76rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); font-weight:600; }
div[data-testid="stMetricValue"]{ font-weight:800; }

.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button{
  border-radius:11px; font-weight:600; border:1px solid var(--line); transition:all .15s ease;
}
.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button[kind="primary"]{
  background:linear-gradient(120deg,#0EA5E9,#6366F1); border:none; color:#fff; box-shadow:0 6px 16px rgba(99,102,241,.35);
}
.stButton > button:hover, .stDownloadButton > button:hover{ transform:translateY(-1px); border-color:var(--acc); }

/* tabs */
button[data-baseweb="tab"]{ font-weight:600; padding:10px 16px; border-radius:10px 10px 0 0; }
div[data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--line); }
div[data-baseweb="tab-highlight"]{ background:linear-gradient(90deg,var(--acc),var(--acc2)) !important; height:3px; border-radius:3px; }

/* expanders, dataframes, alerts, forms, inputs */
details, div[data-testid="stExpander"]{ border:1px solid var(--line) !important; border-radius:14px !important; background:var(--card); }
div[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:12px; overflow:hidden; }
div[data-testid="stAlert"]{ border-radius:12px; }
div[data-testid="stForm"]{ border:1px solid var(--line); border-radius:16px; padding:20px; background:var(--card); }
div[data-baseweb="input"], div[data-baseweb="select"] > div, textarea{ border-radius:10px !important; }
div[data-testid="stVerticalBlockBorderWrapper"]{ border-radius:16px; }
iframe[title="streamlit_folium.st_folium"]{ border-radius:14px; border:1px solid var(--line); }

/* scrollbar */
::-webkit-scrollbar{ width:9px; height:9px; }
::-webkit-scrollbar-thumb{ background:rgba(148,163,184,.35); border-radius:8px; }
</style>
"""


def apply_theme():
    """Inject the global CSS (call once, right after set_page_config)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(page, chips=None):
    """Gradient banner shown at the top of every page."""
    icon, title, subtitle = PAGE_META.get(page, ("🚚", page, ""))
    chip_html = "".join(f'<span class="chip">{escape(str(c))}</span>' for c in (chips or []))
    st.markdown(
        f'<div class="hero"><h1>{icon} {escape(title)}</h1>'
        f'<p>{escape(subtitle)}</p>'
        f'<div class="chips">{chip_html}</div></div>',
        unsafe_allow_html=True,
    )


def section(title, subtitle=""):
    """Accent-bar section heading (replaces plain st.subheader where wanted)."""
    sub = f'<span class="s">{escape(subtitle)}</span>' if subtitle else ""
    st.markdown(
        f'<div class="sec"><div class="bar"></div><div class="t">{escape(title)}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def kpi(col, label, value, icon="", sub="", color="#38BDF8"):
    """Render one KPI card inside the given Streamlit column."""
    with col:
        st.markdown(
            f'<div class="kpi" style="--kc:{color}">'
            f'<div class="k-top"><span class="k-label">{escape(str(label))}</span>'
            f'<span class="k-icon">{icon}</span></div>'
            f'<div class="k-value">{escape(str(value))}</div>'
            f'<div class="k-sub">{escape(str(sub))}</div></div>',
            unsafe_allow_html=True,
        )


def status_badge(status):
    emoji, css = STATUS_STYLE.get(str(status), ("•", "st-grey"))
    return f'<span class="badge {css}">{emoji} {escape(str(status))}</span>'


def brand():
    st.sidebar.markdown(
        '<div class="brand"><div class="logo">🚚</div><div>'
        '<div class="name">AI-Based Supply Chain<br>Disturbance Prediction</div>'
        '<div class="tag">Smart route &amp; risk monitoring</div></div></div>',
        unsafe_allow_html=True,
    )


def nav_label(text):
    st.sidebar.markdown(f'<div class="nav-label">{escape(text)}</div>', unsafe_allow_html=True)


def active_shipment_card(shipment_id, source, destination, status, deadline=""):
    st.sidebar.markdown(
        f'<div class="ship-card"><div class="sc-route">#{shipment_id} · {escape(str(source))} → '
        f'{escape(str(destination))}</div>'
        f'<div class="sc-meta">{status_badge(status)}'
        f'{"  ⏰ " + escape(str(deadline)) if deadline else ""}</div></div>',
        unsafe_allow_html=True,
    )
