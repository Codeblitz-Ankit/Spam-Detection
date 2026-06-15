"""
FilterX — Federated Learning Monitor  v2  (+ FX Chart Modal)
=============================================================
Install dependencies:
  pip3 install streamlit plotly pandas requests --break-system-packages

Deploy:
  streamlit run ~/dashboard.py --server.headless true --server.port 8501
"""

import copy
import streamlit as st
import requests
import time
import pandas as pd
import plotly.graph_objects as go
import random

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FilterX · FL Monitor",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "dark" not in st.session_state:
    st.session_state.dark = True

# ── FX ADDED: modal state ────────────────────────────────────────────────────
if "fx_expanded_chart" not in st.session_state:
    st.session_state.fx_expanded_chart = None
if "fx_expanded_title" not in st.session_state:
    st.session_state.fx_expanded_title = ""
# Figure store — populated during render, read by modal block at bottom
if "fx_figures" not in st.session_state:
    st.session_state.fx_figures = {}
# ── END FX ADDED ─────────────────────────────────────────────────────────────

D = st.session_state.dark

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────
if D:
    BG          = "#0D0F14"
    SURFACE     = "#13161E"
    SURFACE2    = "#1C2030"
    BORDER      = "#252A3A"
    BORDER2     = "#343D55"
    TEXT        = "#DDE3F0"
    TEXT2       = "#8892AA"
    TEXT3       = "#4A5368"
    CHART_BG    = "#13161E"
    GRID_C      = "#1A1F2E"
else:
    BG          = "#ECEEF4"
    SURFACE     = "#FFFFFF"
    SURFACE2    = "#F2F4FA"
    BORDER      = "#D2D7E8"
    BORDER2     = "#B0B8CF"
    TEXT        = "#0C1020"
    TEXT2       = "#3A4560"
    TEXT3       = "#7A85A0"
    CHART_BG    = "#FFFFFF"
    GRID_C      = "#EBEEf7"

BLUE        = "#3B82F6" if D else "#1D4ED8"
GREEN       = "#22C55E" if D else "#15803D"
RED         = "#EF4444" if D else "#B91C1C"
YELLOW      = "#F59E0B" if D else "#B45309"
CYAN        = "#06B6D4" if D else "#0E7490"
PURPLE      = "#A855F7" if D else "#7C3AED"
ORANGE      = "#F97316" if D else "#C2410C"

BLUE_S      = "rgba(59,130,246,0.12)"   if D else "rgba(29,78,216,0.08)"
GREEN_S     = "rgba(34,197,94,0.12)"    if D else "rgba(21,128,61,0.08)"
RED_S       = "rgba(239,68,68,0.12)"    if D else "rgba(185,28,28,0.08)"
YELLOW_S    = "rgba(245,158,11,0.12)"   if D else "rgba(180,83,9,0.08)"
CYAN_S      = "rgba(6,182,212,0.10)"    if D else "rgba(14,116,144,0.07)"
PURPLE_S    = "rgba(168,85,247,0.12)"   if D else "rgba(124,58,237,0.08)"

FONT        = "DM Mono, monospace"
FONT_UI     = "DM Sans, sans-serif"

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — original block unchanged, FX modal CSS appended below it
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

*,*::before,*::after{{box-sizing:border-box;}}

html,body,[class*="css"],.main{{
  font-family:'{FONT_UI}',sans-serif!important;
  background:{BG}!important;
  color:{TEXT}!important;
}}
.block-container{{padding:0!important;max-width:100%!important;}}

/* Kill ALL Streamlit chrome */
header[data-testid="stHeader"]{{display:none!important;}}
#MainMenu{{display:none!important;}}
footer{{display:none!important;}}
[data-testid="collapsedControl"]{{display:none!important;}}
.stDeployButton{{display:none!important;}}
[data-testid="stToolbar"]{{display:none!important;}}
[data-testid="stDecoration"]{{display:none!important;}}
.stStatusWidget{{display:none!important;}}

/* ── NAV ── */
.fx-nav{{
  display:flex;align-items:center;justify-content:space-between;
  padding:0 28px;height:52px;
  background:{SURFACE};border-bottom:2px solid {BORDER};
  position:sticky;top:0;z-index:999;
}}
.fx-brand{{display:flex;align-items:center;gap:12px;}}
.fx-hex{{
  width:30px;height:30px;
  background:linear-gradient(135deg,{BLUE},{PURPLE});
  clip-path:polygon(50% 0%,93% 25%,93% 75%,50% 100%,7% 75%,7% 25%);
  flex-shrink:0;
}}
.fx-wordmark{{
  font-family:'Syne',sans-serif;font-size:17px;font-weight:800;
  color:{TEXT};letter-spacing:-0.5px;
}}
.fx-wordmark em{{color:{BLUE};font-style:normal;}}
.fx-crumb{{
  font-family:'{FONT}';font-size:10px;
  color:{TEXT3};padding:3px 8px;
  border:1px solid {BORDER};border-radius:3px;
}}
.fx-nav-right{{display:flex;align-items:center;gap:16px;}}
.fx-pill{{
  display:inline-flex;align-items:center;gap:6px;
  padding:4px 10px;border-radius:4px;
  font-family:'{FONT}';font-size:10px;font-weight:500;
}}
.fx-pill-on{{background:{GREEN_S};color:{GREEN};border:1px solid {GREEN};}}
.fx-pill-off{{background:{RED_S};color:{RED};border:1px solid {RED};}}
.fx-dot{{width:6px;height:6px;border-radius:50%;background:currentColor;}}
.fx-ts{{font-family:'{FONT}';font-size:10px;color:{TEXT3};}}

/* ── LAYOUT ── */
.fx-wrap{{display:grid;grid-template-columns:210px 1fr;min-height:calc(100vh - 52px);}}
.fx-side{{
  background:{SURFACE};border-right:1px solid {BORDER};
  padding:16px 0 24px 0;
  display:flex;flex-direction:column;gap:1px;
  overflow-y:auto;
}}
.fx-main{{padding:20px 24px 32px 24px;overflow-y:auto;}}

/* ── SIDEBAR KPIs ── */
.fx-kpi{{
  padding:14px 18px;
  border-left:3px solid transparent;
  position:relative;
}}
.fx-kpi::before{{
  content:'';position:absolute;inset:0;
  background:transparent;transition:background .12s;
}}
.fx-kpi:hover::before{{background:{SURFACE2};}}
.fx-kpi.c-blue{{border-left-color:{BLUE};}}
.fx-kpi.c-green{{border-left-color:{GREEN};}}
.fx-kpi.c-red{{border-left-color:{RED};}}
.fx-kpi.c-yellow{{border-left-color:{YELLOW};}}
.fx-kpi.c-cyan{{border-left-color:{CYAN};}}
.fx-kpi.c-purple{{border-left-color:{PURPLE};}}
.fx-kpi.c-orange{{border-left-color:{ORANGE};}}
.fx-kpi-lbl{{
  font-size:9px;font-weight:600;letter-spacing:1.1px;
  text-transform:uppercase;color:{TEXT3};margin-bottom:5px;
  position:relative;z-index:1;
}}
.fx-kpi-val{{
  font-family:'{FONT}';font-size:20px;font-weight:500;
  color:{TEXT};line-height:1;letter-spacing:-0.5px;
  position:relative;z-index:1;
}}
.fx-kpi-val .u{{font-size:11px;font-weight:400;color:{TEXT3};margin-left:2px;}}
.fx-kpi-sub{{
  font-family:'{FONT}';font-size:10px;margin-top:4px;
  position:relative;z-index:1;
}}
.pos{{color:{GREEN};}} .neg{{color:{RED};}} .neu{{color:{TEXT3};}}
.yel{{color:{YELLOW};}} .blu{{color:{BLUE};}} .cyn{{color:{CYAN};}}
.pur{{color:{PURPLE};}}

.fx-sdiv{{height:1px;background:{BORDER};margin:10px 18px;}}
.fx-slbl{{
  padding:8px 18px 3px 18px;
  font-size:9px;font-weight:700;letter-spacing:1.5px;
  text-transform:uppercase;color:{TEXT3};
}}

/* ── SECTION HEADERS ── */
.fx-sec{{
  display:flex;align-items:center;gap:9px;
  margin:20px 0 10px 0;
}}
.fx-sec-bar{{width:3px;height:13px;border-radius:2px;flex-shrink:0;}}
.fx-sec-ttl{{
  font-size:10px;font-weight:700;letter-spacing:1.1px;
  text-transform:uppercase;color:{TEXT2};
}}

/* ── CARDS ── */
.fx-card{{
  background:{SURFACE};border:1px solid {BORDER};
  border-radius:6px;overflow:hidden;margin-bottom:16px;
}}
.fx-card-hdr{{
  display:flex;align-items:center;justify-content:space-between;
  padding:11px 16px 9px 16px;border-bottom:1px solid {BORDER};
}}
.fx-card-ttl{{
  font-size:10px;font-weight:700;letter-spacing:1px;
  text-transform:uppercase;color:{TEXT2};
}}
.fx-card-badge{{
  font-family:'{FONT}';font-size:9px;color:{TEXT3};
  background:{SURFACE2};border:1px solid {BORDER};
  padding:2px 7px;border-radius:3px;
}}
.fx-strip{{
  display:flex;gap:0;border-bottom:1px solid {BORDER};
}}
.fx-strip-item{{
  padding:8px 14px;flex:1;
  border-right:1px solid {BORDER};
}}
.fx-strip-item:last-child{{border-right:none;}}
.fx-strip-lbl{{font-size:9px;font-weight:600;letter-spacing:.9px;text-transform:uppercase;color:{TEXT3};}}
.fx-strip-val{{font-family:'{FONT}';font-size:12px;font-weight:500;margin-top:2px;}}

/* ── LOG TERMINAL ── */
.fx-log{{
  background:{"#080B10" if D else "#F5F7FC"};
  padding:10px 14px;height:148px;
  overflow-y:auto;
  font-family:'{FONT}';font-size:11px;line-height:1.75;
}}
.l-ts{{color:{TEXT3};}}
.l-ok{{color:{GREEN};}}
.l-info{{color:{CYAN};}}
.l-warn{{color:{YELLOW};}}
.l-err{{color:{RED};}}
.l-data{{color:{PURPLE};}}
.l-msg{{color:{TEXT2};}}

/* ── TABLE ── */
[data-testid="stDataFrame"] table{{
  font-family:'{FONT}'!important;font-size:11px!important;
  color:{TEXT}!important;
}}
[data-testid="stDataFrame"] th{{
  background:{SURFACE2}!important;color:{TEXT2}!important;
  font-size:9px!important;letter-spacing:.9px!important;
  text-transform:uppercase!important;
  border-bottom:1px solid {BORDER2}!important;
  padding:8px 12px!important;
}}
[data-testid="stDataFrame"] td{{
  color:{TEXT}!important;border-bottom:1px solid {BORDER}!important;
  padding:7px 12px!important;
}}

/* ── EMPTY ── */
.fx-empty{{
  padding:44px 28px;text-align:center;
  color:{TEXT3};font-size:12px;
  background:{SURFACE};border:1px dashed {BORDER2};border-radius:6px;
}}
.fx-empty h4{{
  font-family:'Syne',sans-serif;font-size:15px;
  font-weight:700;color:{TEXT2};margin-bottom:8px;
}}

/* ── THEME BTN ── */
[data-testid="stButton"] button{{
  font-family:'{FONT}'!important;font-size:10px!important;
  font-weight:500!important;border-radius:4px!important;
  padding:5px 12px!important;
  background:{SURFACE2}!important;border:1px solid {BORDER2}!important;
  color:{TEXT2}!important;white-space:nowrap;
}}
[data-testid="stButton"] button:hover{{
  background:{BORDER}!important;color:{TEXT}!important;
}}

/* ── FOOTER ── */
.fx-foot{{
  display:flex;justify-content:space-between;align-items:center;
  padding:10px 28px;
  border-top:1px solid {BORDER};background:{SURFACE};
}}
.fx-foot span{{font-family:'{FONT}';font-size:10px;color:{TEXT3};}}

/* Grid helpers */
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
.g3{{display:grid;grid-template-columns:2fr 1fr 2fr;gap:16px;}}
.g32{{display:grid;grid-template-columns:3fr 2fr;gap:16px;}}
</style>
""", unsafe_allow_html=True)

# ── FX ADDED: modal + typography CSS ─────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

/* ── Poppins on labels, table, metric values ── */
[data-testid="stMetricValue"] {{
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}}
[data-testid="stMetricLabel"] {{
    font-family: 'Poppins', sans-serif !important;
    font-weight: 400 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-size: 0.68rem !important;
    opacity: 0.72 !important;
}}
[data-testid="stDataFrame"] th {{
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    font-size: 0.7rem !important;
}}
[data-testid="stDataFrame"] td {{
    font-family: 'Poppins', sans-serif !important;
    font-weight: 400 !important;
    font-size: 0.82rem !important;
}}

/* ── Expand button — tiny, right-aligned, unobtrusive ── */
.fx-expand-row {{
    display: flex;
    justify-content: flex-end;
    padding: 2px 8px 6px 0;
}}
button[data-testid="baseButton-secondary"].fx-expand-trigger {{
    background: transparent !important;
    border: 1px solid {BORDER2} !important;
    color: {TEXT3} !important;
    font-size: 11px !important;
    padding: 2px 8px !important;
    border-radius: 3px !important;
    min-height: unset !important;
    height: 22px !important;
    line-height: 1 !important;
    transition: border-color 0.15s, color 0.15s !important;
}}
button[data-testid="baseButton-secondary"].fx-expand-trigger:hover {{
    border-color: {BLUE} !important;
    color: {BLUE} !important;
    background: {BLUE_S} !important;
}}

/* ── Modal overlay — fixed, full-screen, blurred backdrop ── */
.fx-modal-overlay {{
    position: fixed !important;
    inset: 0 !important;
    z-index: 999998 !important;
    background: rgba(0, 0, 0, 0.75) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 24px !important;
    box-sizing: border-box !important;
    animation: fxFadeIn 0.2s ease forwards !important;
}}

/* ── Glassmorphism card ── */
.fx-modal-glass {{
    width: 88vw;
    max-width: 1300px;
    background: rgba(15, 18, 28, 0.82) !important;
    backdrop-filter: blur(32px) saturate(160%) !important;
    -webkit-backdrop-filter: blur(32px) saturate(160%) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 18px !important;
    box-shadow:
        0 16px 48px rgba(0, 0, 0, 0.7),
        0 0 0 1px rgba(255,255,255,0.03) inset !important;
    overflow: hidden;
    animation: fxScaleIn 0.26s cubic-bezier(0.34, 1.4, 0.64, 1) forwards;
}}

/* ── Modal header bar ── */
.fx-modal-hdr {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 15px 22px 13px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}}
.fx-modal-ttl {{
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.5) !important;
    margin: 0;
}}

/* ── Close button — fixed top-right of modal ── */
.fx-modal-close-wrap {{
    position: fixed !important;
    top: calc(50vh - 40vh + 14px) !important;
    right: calc(6vw + 14px) !important;
    z-index: 999999 !important;
}}
.fx-modal-close-wrap button {{
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    color: rgba(255,255,255,0.65) !important;
    border-radius: 7px !important;
    font-size: 14px !important;
    width: 34px !important;
    height: 34px !important;
    padding: 0 !important;
    min-height: unset !important;
    line-height: 1 !important;
    cursor: pointer !important;
    font-family: 'Poppins', sans-serif !important;
    transition: background 0.15s, color 0.15s !important;
}}
.fx-modal-close-wrap button:hover {{
    background: rgba(255,255,255,0.16) !important;
    color: #fff !important;
}}

/* ── Plotly chart inside modal ── */
.fx-modal-body {{
    padding: 16px 20px 20px;
}}
.fx-modal-body [data-testid="stPlotlyChart"] {{
    border-radius: 8px;
    overflow: hidden;
}}

/* ── Animations ── */
@keyframes fxFadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
@keyframes fxScaleIn {{
    from {{ opacity: 0; transform: scale(0.93) translateY(8px); }}
    to   {{ opacity: 1; transform: scale(1)    translateY(0);   }}
}}

/* ── Responsive ── */
@media (max-width: 768px) {{
    .fx-modal-glass  {{ width: 97vw !important; border-radius: 13px !important; }}
    .fx-modal-hdr    {{ padding: 12px 16px 10px; }}
    .fx-modal-body   {{ padding: 10px 12px 14px; }}
    .fx-modal-close-wrap {{ top: 12px !important; right: 12px !important; }}
}}
</style>
""", unsafe_allow_html=True)
# ── END FX ADDED ──────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
SERVER = "http://localhost:8000"

def fetch():
    try:
        r = requests.get(f"{SERVER}/metrics", timeout=3)
        return r.json(), True
    except:
        return None, False

data, online = fetch()
ts    = time.strftime('%H:%M:%S')
ts_dt = time.strftime('%d %b %Y · %H:%M:%S')

if online and data:
    current_round = data.get('round', 0)
    current_acc   = data.get('accuracy', 0)
    pending       = data.get('pending_uploads', 0)
    history       = data.get('history', [])
else:
    current_round, current_acc, pending, history = 0, 0, 0, []

df           = pd.DataFrame(history) if history else pd.DataFrame()
last_delta   = float(df['delta'].iloc[-1]) if not df.empty else None
total_cli    = int(df['num_clients'].sum()) if not df.empty else 0
total_samp   = int(df['samples'].sum()) if not df.empty else 0
best_acc     = float(df['new_accuracy'].max()) if not df.empty else current_acc
avg_delta    = float(df['delta'].mean()) if not df.empty else 0.0
pos_rounds   = int((df['delta'] > 0).sum()) if not df.empty else 0
n_rounds     = len(history)

rng = random.Random(42 + current_round)
n_pts = max(24, n_rounds * 10)
lat_pts = []
for i in range(n_pts):
    base = 115 + rng.gauss(0, 18)
    if i % 10 == 9:
        base += rng.uniform(60, 120)
    lat_pts.append(max(30, round(base, 1)))

def mk_log():
    lines = []
    if not online:
        lines.append(('err', ts, 'CONN', f'connection refused — {SERVER}'))
        return lines
    for h in history[-10:]:
        r = h['round']
        lines.append(('data', f"R{r}", 'FEDAVG',
                       f"aggregated {h.get('num_clients',3)} clients · {h.get('samples',0):,} samples"))
        sign = '+' if h['delta'] >= 0 else ''
        cls  = 'ok' if h['delta'] >= 0 else 'err'
        lines.append((cls, f"R{r}", 'MODEL',
                       f"acc {h['old_accuracy']:.4f}% → {h['new_accuracy']:.4f}%  (Δ {sign}{h['delta']:.4f}%)"))
    for i in range(pending):
        lines.append(('info', ts, 'UPLOAD', f"client #{i+1} received · waiting for {3-pending} more"))
    lines.append(('info', ts, 'SERVER',
                  f"online · round {current_round} · {pending}/3 uploads pending"))
    return lines[-16:]

log_entries = mk_log()

# ─────────────────────────────────────────────────────────────────────────────
# NAV BAR
# ─────────────────────────────────────────────────────────────────────────────
pill_cls = "fx-pill-on" if online else "fx-pill-off"
pill_txt = "● CONNECTED" if online else "● OFFLINE"

nav_c, btn_c = st.columns([15, 1])
with nav_c:
    st.markdown(f"""
    <div class="fx-nav">
      <div class="fx-brand">
        <div class="fx-hex"></div>
        <div class="fx-wordmark">Filter<em>X</em></div>
        <span class="fx-crumb">FL Monitor</span>
      </div>
      <div class="fx-nav-right">
        <div class="fx-pill {pill_cls}"><div class="fx-dot"></div>{pill_txt}</div>
        <span class="fx-ts">{ts_dt}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
with btn_c:
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    if st.button("☀ Light" if D else "◑ Dark", key="thm"):
        st.session_state.dark = not D
        st.rerun()

if not online or not data:
    st.markdown(f"""
    <div style="padding:60px 32px;">
      <div class="fx-empty">
        <h4>Server Unreachable</h4>
        Cannot reach <code style="font-family:'DM Mono',monospace;
          background:{SURFACE2};padding:2px 8px;border-radius:3px;
          border:1px solid {BORDER2}">{SERVER}</code><br><br>
        Ensure uvicorn is running. Retrying in 5 s…
      </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(5)
    st.rerun()
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  +  MAIN
# ─────────────────────────────────────────────────────────────────────────────
side_col, main_col = st.columns([1, 4], gap="small")

# ── FX ADDED: helper — renders a tiny expand button below a chart ─────────────
def _fx_expand_btn(key: str, title: str):
    """Render the ⤢ expand button. On click, sets modal state and reruns."""
    col_gap, col_btn = st.columns([20, 1])
    with col_btn:
        if st.button("⤢", key=f"fx_exp_{key}", help="Expand chart"):
            st.session_state.fx_expanded_chart = key
            st.session_state.fx_expanded_title = title
            st.rerun()
# ── END FX ADDED ─────────────────────────────────────────────────────────────

# ══════════════════ SIDEBAR ══════════════════
with side_col:
    dp   = f"+{last_delta:.4f}" if last_delta and last_delta >= 0 else (f"{last_delta:.4f}" if last_delta else "—")
    dsub = ("pos", "▲ improving") if (last_delta or 0) >= 0 else ("neg", "▼ degraded")
    psub = "yel" if 0 < pending < 3 else ("neu" if pending == 0 else "pos")

    st.markdown(f"""
    <div class="fx-side">
      <div class="fx-slbl">Training State</div>

      <div class="fx-kpi c-blue">
        <div class="fx-kpi-lbl">FL Round</div>
        <div class="fx-kpi-val">{current_round}<span class="u"> rnd</span></div>
        <div class="fx-kpi-sub neu">{n_rounds} completed</div>
      </div>

      <div class="fx-kpi c-green">
        <div class="fx-kpi-lbl">Model Accuracy</div>
        <div class="fx-kpi-val">{current_acc:.2f}<span class="u">%</span></div>
        <div class="fx-kpi-sub neu">Best {best_acc:.4f}%</div>
      </div>

      <div class="fx-kpi c-{'green' if (last_delta or 0)>=0 else 'red'}">
        <div class="fx-kpi-lbl">Last Δ Accuracy</div>
        <div class="fx-kpi-val">{dp}<span class="u">%</span></div>
        <div class="fx-kpi-sub {dsub[0]}">{dsub[1]}</div>
      </div>

      <div class="fx-sdiv"></div>
      <div class="fx-slbl">Aggregation</div>

      <div class="fx-kpi c-yellow">
        <div class="fx-kpi-lbl">Pending Uploads</div>
        <div class="fx-kpi-val">{pending}<span class="u"> / 3</span></div>
        <div class="fx-kpi-sub {psub}">{3-pending} until FedAvg</div>
      </div>

      <div class="fx-kpi c-cyan">
        <div class="fx-kpi-lbl">Total Clients</div>
        <div class="fx-kpi-val">{total_cli}<span class="u"> cli</span></div>
        <div class="fx-kpi-sub neu">{total_samp:,} samples</div>
      </div>

      <div class="fx-sdiv"></div>
      <div class="fx-slbl">Analytics</div>

      <div class="fx-kpi c-purple">
        <div class="fx-kpi-lbl">Avg Δ / Round</div>
        <div class="fx-kpi-val">{avg_delta:+.4f}<span class="u">%</span></div>
        <div class="fx-kpi-sub {'pos' if avg_delta>=0 else 'neg'}">{'▲ pos trend' if avg_delta>=0 else '▼ neg trend'}</div>
      </div>

      <div class="fx-kpi c-orange">
        <div class="fx-kpi-lbl">Positive Rounds</div>
        <div class="fx-kpi-val">{pos_rounds}<span class="u"> / {n_rounds}</span></div>
        <div class="fx-kpi-sub neu">{'—' if n_rounds==0 else f'{pos_rounds/n_rounds*100:.0f}% success'}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════ MAIN ══════════════════
with main_col:

    BASE = dict(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(family=FONT, color=TEXT3, size=10),
        margin=dict(l=8, r=8, t=6, b=6),
        xaxis=dict(showgrid=True, gridcolor=GRID_C, gridwidth=1,
                   zeroline=False, showline=False, color=TEXT3,
                   tickfont=dict(size=9, family=FONT, color=TEXT3)),
        yaxis=dict(showgrid=True, gridcolor=GRID_C, gridwidth=1,
                   zeroline=False, showline=False, color=TEXT3,
                   tickfont=dict(size=9, family=FONT, color=TEXT3)),
        hoverlabel=dict(bgcolor=SURFACE2, bordercolor=BORDER2,
                        font=dict(family=FONT, size=11, color=TEXT)),
        showlegend=False,
    )

    # ── ROW 1 ────────────────────────────────────────────────────────────────
    if not df.empty:
        acc_pts = [{"r": 0, "a": history[0]["old_accuracy"]}] + \
                  [{"r": h["round"], "a": h["new_accuracy"]} for h in history]
        adf   = pd.DataFrame(acc_pts)

        c1, c2 = st.columns(2, gap="small")

        with c1:
            st.markdown(f"""
            <div class="fx-card">
              <div class="fx-card-hdr">
                <div class="fx-card-ttl">Accuracy · Round over Round</div>
                <div class="fx-card-badge">{current_acc:.4f}%</div>
              </div>
              <div class="fx-strip">
                <div class="fx-strip-item">
                  <div class="fx-strip-lbl">Current</div>
                  <div class="fx-strip-val blu">{current_acc:.4f}%</div>
                </div>
                <div class="fx-strip-item">
                  <div class="fx-strip-lbl">Best</div>
                  <div class="fx-strip-val pos">{best_acc:.4f}%</div>
                </div>
                <div class="fx-strip-item">
                  <div class="fx-strip-lbl">Rounds</div>
                  <div class="fx-strip-val cyn">{n_rounds}</div>
                </div>
              </div>
            """, unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=adf["r"], y=adf["a"], fill="tozeroy",
                fillcolor=BLUE_S, line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=adf["r"], y=adf["a"], mode="lines+markers",
                line=dict(color=BLUE, width=2.5),
                marker=dict(color=BLUE, size=6, line=dict(width=2, color=CHART_BG)),
                hovertemplate="Round %{x}<br><b>%{y:.4f}%</b><extra></extra>"
            ))
            fig.update_layout(**BASE, height=195)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.session_state.fx_figures["accuracy_round"] = fig  # ── FX ADDED
            _fx_expand_btn("accuracy_round", "Accuracy · Round over Round")  # ── FX ADDED
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            bar_colors = [GREEN if d >= 0 else RED for d in df["delta"]]
            st.markdown(f"""
            <div class="fx-card">
              <div class="fx-card-hdr">
                <div class="fx-card-ttl">Accuracy Delta · Per Round</div>
                <div class="fx-card-badge">{dp}%</div>
              </div>
              <div class="fx-strip">
                <div class="fx-strip-item">
                  <div class="fx-strip-lbl">Last Δ</div>
                  <div class="fx-strip-val {'pos' if (last_delta or 0)>=0 else 'neg'}">{dp}%</div>
                </div>
                <div class="fx-strip-item">
                  <div class="fx-strip-lbl">Avg Δ</div>
                  <div class="fx-strip-val yel">{avg_delta:+.4f}%</div>
                </div>
                <div class="fx-strip-item">
                  <div class="fx-strip-lbl">Positive</div>
                  <div class="fx-strip-val pos">{pos_rounds}/{n_rounds}</div>
                </div>
              </div>
            """, unsafe_allow_html=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=df["round"], y=df["delta"],
                marker=dict(color=bar_colors, opacity=0.88, line=dict(width=0)),
                hovertemplate="Round %{x}<br><b>Δ %{y:+.4f}%</b><extra></extra>"
            ))
            fig2.add_hline(y=0, line_color=BORDER2, line_width=1)
            fig2.update_layout(**BASE, height=195, bargap=0.3)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            st.session_state.fx_figures["accuracy_delta"] = fig2  # ── FX ADDED
            _fx_expand_btn("accuracy_delta", "Accuracy Delta · Per Round")  # ── FX ADDED
            st.markdown("</div>", unsafe_allow_html=True)

    # ── ROW 2 ────────────────────────────────────────────────────────────────
    c3, c4, c5 = st.columns([2, 1, 2], gap="small")

    with c3:
        st.markdown(f"""
        <div class="fx-card">
          <div class="fx-card-hdr">
            <div class="fx-card-ttl">Cumulative Accuracy Gain</div>
            <div class="fx-card-badge">Σ Δ rounds</div>
          </div>
        """, unsafe_allow_html=True)
        if not df.empty:
            df2 = df.copy()
            df2["cum"] = df2["delta"].cumsum()
            cc = [GREEN if v >= 0 else RED for v in df2["cum"]]
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=df2["round"], y=df2["cum"],
                marker=dict(color=cc, opacity=0.85, line=dict(width=0)),
                hovertemplate="Round %{x}<br><b>Σ Δ %{y:+.4f}%</b><extra></extra>"
            ))
            fig3.add_hline(y=0, line_color=BORDER2, line_width=1)
            fig3.update_layout(**BASE, height=155, bargap=0.3)
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
            st.session_state.fx_figures["cumulative_gain"] = fig3  # ── FX ADDED
            _fx_expand_btn("cumulative_gain", "Cumulative Accuracy Gain")  # ── FX ADDED
        else:
            st.markdown(f'<div style="height:155px;display:flex;align-items:center;justify-content:center;color:{TEXT3};font-family:{FONT};font-size:11px;">Awaiting first round</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        gc = YELLOW if pending < 3 else GREEN
        st.markdown(f"""
        <div class="fx-card">
          <div class="fx-card-hdr">
            <div class="fx-card-ttl">Upload Status</div>
          </div>
        """, unsafe_allow_html=True)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pending,
            number=dict(font=dict(family=FONT, color=TEXT, size=26), suffix=" / 3"),
            gauge=dict(
                axis=dict(range=[0, 3], tickfont=dict(family=FONT, size=8, color=TEXT3),
                          nticks=4, tickcolor=TEXT3),
                bar=dict(color=gc, thickness=0.65),
                bgcolor=SURFACE2,
                bordercolor=BORDER, borderwidth=1,
                steps=[
                    dict(range=[0, 1], color=RED_S),
                    dict(range=[1, 2], color=YELLOW_S),
                    dict(range=[2, 3], color=GREEN_S),
                ],
                threshold=dict(line=dict(color=GREEN, width=2), thickness=0.85, value=3)
            )
        ))
        fig_g.update_layout(
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            font=dict(family=FONT, color=TEXT),
            margin=dict(l=10, r=10, t=14, b=0), height=140
        )
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})
        st.session_state.fx_figures["upload_status"] = fig_g  # ── FX ADDED
        _fx_expand_btn("upload_status", "Upload Status")  # ── FX ADDED
        need = 3 - pending
        sub_c = YELLOW if need > 0 else GREEN
        sub_t = f"{need} more for FedAvg" if need > 0 else "Ready to aggregate"
        st.markdown(f'<div style="text-align:center;padding:4px 0 10px;font-family:{FONT};font-size:10px;color:{sub_c};">{sub_t}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="fx-card">
          <div class="fx-card-hdr">
            <div class="fx-card-ttl">Client Upload Latency</div>
            <div class="fx-card-badge">simulated · ms</div>
          </div>
        """, unsafe_allow_html=True)
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(
            x=list(range(len(lat_pts))), y=lat_pts,
            mode="lines", line=dict(color=CYAN, width=1.5),
            fill="tozeroy", fillcolor=CYAN_S,
            hovertemplate="t=%{x}<br><b>%{y:.0f} ms</b><extra></extra>"
        ))
        fig_l.add_hline(y=180, line_color=RED, line_width=1, line_dash="dot",
                        annotation_text="threshold", annotation_font_color=RED,
                        annotation_font_size=8, annotation_position="top right")
        fig_l.update_layout(**BASE, height=155)
        st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar": False})
        st.session_state.fx_figures["client_latency"] = fig_l  # ── FX ADDED
        _fx_expand_btn("client_latency", "Client Upload Latency")  # ── FX ADDED
        st.markdown("</div>", unsafe_allow_html=True)

    # ── ROW 3 ────────────────────────────────────────────────────────────────
    c6, c7 = st.columns([3, 2], gap="small")

    with c6:
        st.markdown(f"""
        <div class="fx-card">
          <div class="fx-card-hdr">
            <div class="fx-card-ttl">Activity Log</div>
            <div class="fx-card-badge" style="color:{GREEN};">● Live</div>
          </div>
          <div class="fx-log">
        """, unsafe_allow_html=True)
        log_html = ""
        for (kind, t, tag, msg) in reversed(log_entries):
            tc = {"data":"l-data","ok":"l-ok","info":"l-info","warn":"l-warn","err":"l-err"}.get(kind,"l-ts")
            log_html += f"""<div style="display:flex;gap:0;align-items:baseline;padding:2px 0;border-bottom:1px solid rgba(255,255,255,0.03);">
                <span class="l-ts" style="min-width:72px;flex-shrink:0;">[{t}]</span>
                <span class="{tc}" style="min-width:72px;flex-shrink:0;font-weight:500;">[{tag}]</span>
                <span class="l-msg" style="flex:1;">{msg}</span>
            </div>"""
        st.markdown(log_html + "</div></div>", unsafe_allow_html=True)

    with c7:
        st.markdown(f"""
        <div class="fx-card">
          <div class="fx-card-hdr">
            <div class="fx-card-ttl">Samples per Round</div>
            <div class="fx-card-badge">{total_samp:,} total</div>
          </div>
        """, unsafe_allow_html=True)
        if not df.empty:
            fig_s = go.Figure()
            fig_s.add_trace(go.Bar(
                x=df["round"], y=df["samples"],
                marker=dict(color=PURPLE, opacity=0.8, line=dict(width=0)),
                hovertemplate="Round %{x}<br><b>%{y:,} samples</b><extra></extra>"
            ))
            fig_s.update_layout(**BASE, height=148, bargap=0.3)
            st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})
            st.session_state.fx_figures["samples_per_round"] = fig_s  # ── FX ADDED
            _fx_expand_btn("samples_per_round", "Samples per Round")  # ── FX ADDED
        else:
            st.markdown(f'<div style="height:148px;display:flex;align-items:center;justify-content:center;color:{TEXT3};font-family:{FONT};font-size:11px;">No data yet</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── ROW 4: Round history table ────────────────────────────────────────────
    st.markdown(f"""
    <div class="fx-sec">
      <div class="fx-sec-bar" style="background:{BLUE}"></div>
      <div class="fx-sec-ttl">Round History</div>
    </div>
    """, unsafe_allow_html=True)

    if not df.empty:
        st.markdown('<div class="fx-card">', unsafe_allow_html=True)
        disp = pd.DataFrame({
            "Round":       df["round"].astype(int),
            "Old Acc (%)": df["old_accuracy"].map("{:.4f}".format),
            "New Acc (%)": df["new_accuracy"].map("{:.4f}".format),
            "Δ Accuracy":  df["delta"].map("{:+.4f}%".format),
            "Clients":     df["num_clients"].astype(int),
            "Samples":     df["samples"].astype(int).map("{:,}".format),
        })
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="fx-empty">
          <h4>No Rounds Completed</h4>
          Waiting for {3-pending} more upload{'s' if (3-pending)!=1 else ''} to trigger FedAvg.<br>
          Currently {pending}/3 uploads received this round.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="fx-foot">
  <span>FilterX Federated Learning · {SERVER} · Round {current_round}</span>
  <span>Auto-refresh every 3 s · Last sync {ts}</span>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# FX CHART MODAL  — top-level, outside all columns/containers
# Reads from st.session_state.fx_figures populated during the render above.
# ═════════════════════════════════════════════════════════════════════════════
_fx_key = st.session_state.get("fx_expanded_chart")

if _fx_key:
    _fx_title  = st.session_state.get("fx_expanded_title", _fx_key.replace("_", " ").title())
    _fx_fig_orig = st.session_state.fx_figures.get(_fx_key)

    if _fx_fig_orig is not None:
        # Deep-copy so we never mutate the original figure object
        _fx_fig = copy.deepcopy(_fx_fig_orig)
        _fx_fig.update_layout(
            height=560,
            margin=dict(l=44, r=44, t=32, b=44),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # Glassmorphism overlay + header HTML
        st.markdown(f"""
        <div class="fx-modal-overlay" id="fx-overlay">
          <div class="fx-modal-glass">
            <div class="fx-modal-hdr">
              <p class="fx-modal-ttl">{_fx_title}</p>
            </div>
            <div class="fx-modal-body">
        """, unsafe_allow_html=True)

        # The real interactive Plotly chart — full interactions preserved
        st.plotly_chart(_fx_fig, use_container_width=True,
                        key="fx_modal_render",
                        config={"displayModeBar": True,
                                "modeBarButtonsToRemove": ["toImage"],
                                "scrollZoom": True})

        st.markdown("</div></div></div>", unsafe_allow_html=True)

        # Close button — CSS positions it fixed inside the modal header area
        st.markdown('<div class="fx-modal-close-wrap">', unsafe_allow_html=True)
        if st.button("✕", key="fx_close_btn"):
            st.session_state.fx_expanded_chart = None
            st.session_state.fx_expanded_title = ""
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # Minimal JS — only for ESC and backdrop click; touches no chart data
        st.markdown("""
        <script>
        (function(){
            // ESC key closes modal
            function onEsc(e){
                if(e.key!=='Escape') return;
                var btn=document.querySelector('button[data-testid="baseButton-secondary"]');
                if(btn) btn.click();
                document.removeEventListener('keydown',onEsc);
            }
            document.addEventListener('keydown',onEsc);

            // Click on the dark overlay backdrop closes modal
            var ov=document.getElementById('fx-overlay');
            if(ov){
                ov.addEventListener('click',function(e){
                    if(e.target===ov){
                        var btn=document.querySelector('button[data-testid="baseButton-secondary"]');
                        if(btn) btn.click();
                    }
                });
            }
        })();
        </script>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH  (only when no modal is open — avoids rerun interrupting modal)
# ─────────────────────────────────────────────────────────────────────────────
if not _fx_key:
    time.sleep(3)
    st.rerun()
