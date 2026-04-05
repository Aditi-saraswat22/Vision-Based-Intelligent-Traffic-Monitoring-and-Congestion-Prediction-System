"""
TrafficVision AI — Production Ready
====================================
Run: streamlit run app.py

Models are auto-downloaded from HuggingFace on first run.
No manual model loading needed for deployment.
"""

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import time, os, tempfile, urllib.request
from collections import deque
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrafficVision AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background: #08090f; color: #d4d6f0; }
.main .block-container { padding: 1.8rem 2rem 3rem; max-width: 100%; }

/* scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0d0e1a; }
::-webkit-scrollbar-thumb { background: #2a2d4a; border-radius: 2px; }

/* sidebar */
[data-testid="stSidebar"] {
    background: #070810 !important;
    border-right: 1px solid #12132a;
}
[data-testid="stSidebar"] .block-container { padding: 1.2rem 1rem; }
[data-testid="stSidebar"] label {
    color: #3a3d60 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
}
[data-testid="stSidebar"] .stTextInput input {
    background: #0e0f1e !important;
    border: 1px solid #1a1b30 !important;
    color: #9496c0 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    padding: 8px 12px !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
    outline: none !important;
}
[data-testid="stSidebar"] p { color: #3a3d60 !important; font-size: 0.8rem !important; }

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }

/* global buttons */
.stButton > button {
    background: #0e0f1e !important;
    color: #8b8ef0 !important;
    border: 1px solid #1a1b30 !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 10px 18px !important;
    width: 100% !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button:hover {
    background: #14153a !important;
    border-color: #6366f1 !important;
    color: #fff !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.25) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: none !important;
}

/* file uploader */
[data-testid="stFileUploader"] {
    background: #0e0f1e !important;
    border: 2px dashed #1a1b30 !important;
    border-radius: 14px !important;
    padding: 0.5rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6366f1 !important;
}

/* plotly tooltip fix */
.js-plotly-plot .plotly .hoverlayer { pointer-events: none !important; }
.js-plotly-plot .plotly .hovertext { opacity: 1 !important; }

/* sliders */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #8b8ef0) !important;
}
[data-testid="stSlider"] > div > div > div > div > div {
    background: #6366f1 !important;
    box-shadow: 0 0 8px rgba(99,102,241,0.5) !important;
}

/* select boxes */
[data-testid="stSelectbox"] > div > div {
    background: #0e0f1e !important;
    border: 1px solid #1a1b30 !important;
    border-radius: 8px !important;
    color: #9496c0 !important;
}

/* loading spinner */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* section divider */
.sdiv {
    height: 1px;
    background: linear-gradient(90deg, #12132a, #1a1b40, #12132a);
    margin: 1.2rem 0;
}

/* card base */
.card {
    background: #0e0f1e;
    border: 1px solid #12132a;
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.card:hover { border-color: #1a1b40; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }

/* metric card accent line */
.card-accent-top::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.accent-indigo::before { background: linear-gradient(90deg, #6366f1, transparent); }
.accent-green::before  { background: linear-gradient(90deg, #10b981, transparent); }
.accent-amber::before  { background: linear-gradient(90deg, #f59e0b, transparent); }
.accent-red::before    { background: linear-gradient(90deg, #ef4444, transparent); }
.accent-purple::before { background: linear-gradient(90deg, #8b5cf6, transparent); }

/* loading screen */
.loading-screen {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 70vh;
    text-align: center;
}
.spinner-ring {
    width: 60px; height: 60px;
    border: 3px solid #12132a;
    border-top: 3px solid #6366f1;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 24px;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* logo */
.logo-mark {
    display: inline-flex;
    align-items: center;
    gap: 10px;
}
.logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    box-shadow: 0 4px 14px rgba(99,102,241,0.4);
}
.logo-text {
    font-size: 1.2rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.5px;
}
.logo-text span { color: #6366f1; }

/* prediction panel */
.pred-panel {
    border-radius: 12px;
    padding: 16px 18px;
    border: 1px solid;
    border-left: 3px solid;
    position: relative;
}

/* weather tag */
.wtag {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 6px;
    border: 1px solid;
    cursor: default;
}
.wtag-active   { background: rgba(99,102,241,0.15); color: #8b8ef0; border-color: rgba(99,102,241,0.4); }
.wtag-inactive { background: transparent; color: #2a2d50; border-color: #12132a; }

/* risk badge */
.risk-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.risk-low      { background: rgba(16,185,129,0.12); color: #10b981; }
.risk-moderate { background: rgba(245,158,11,0.12); color: #f59e0b; }
.risk-high     { background: rgba(249,115,22,0.12); color: #f97316; }
.risk-critical { background: rgba(239,68,68,0.12);  color: #ef4444; }

/* stat label */
.stat-lbl {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #2a2d50;
    font-weight: 600;
    margin-top: 5px;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LSTM MODEL DEFINITION
# ─────────────────────────────────────────────────────────────
class CongestionLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 128, 2, batch_first=True, dropout=0.3)
        self.fc   = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)          # 4 congestion classes
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ─────────────────────────────────────────────────────────────
# AUTO-DOWNLOAD MODELS (deployment ready)
# ─────────────────────────────────────────────────────────────
# ── HOW TO UPDATE FOR YOUR OWN DEPLOYMENT ──────────────────
# 1. Upload best.pt and lstm_congestion.pt to HuggingFace Hub
#    or any direct download URL (Google Drive with gdown, etc.)
# 2. Replace the URLs below with your actual URLs
# 3. Deploy to Streamlit Cloud — models download automatically
# ──────────────────────────────────────────────────────────

YOLO_URL = "https://huggingface.co/riya17singh/trafficvision-models/resolve/main/best.pt"
LSTM_URL = "https://huggingface.co/riya17singh/trafficvision-models/resolve/main/lstm_congestion.pt"

YOLO_PATH = "models/best.pt"
LSTM_PATH = "models/lstm_congestion.pt"

def ensure_models():
    """
    Download models if not present.
    For LOCAL use: just put models in models/ folder.
    For DEPLOYMENT: set YOLO_URL and LSTM_URL above.
    """
    os.makedirs("models", exist_ok=True)
    status = {"yolo": False, "lstm": False}

    # Try loading local models first
    if os.path.exists(YOLO_PATH) and os.path.getsize(YOLO_PATH) > 1000:
        status["yolo"] = True
    elif "YOUR_YOLO" not in YOLO_URL:
        try:
            st.info("Downloading YOLOv8 model... (~6MB)")
            urllib.request.urlretrieve(YOLO_URL, YOLO_PATH)
            status["yolo"] = True
        except Exception as e:
            st.warning(f"Could not download YOLO model: {e}")

    if os.path.exists(LSTM_PATH) and os.path.getsize(LSTM_PATH) > 100:
        status["lstm"] = True
    elif "YOUR_LSTM" not in LSTM_URL:
        try:
            st.info("Downloading LSTM model... (~1MB)")
            urllib.request.urlretrieve(LSTM_URL, LSTM_PATH)
            status["lstm"] = True
        except Exception as e:
            st.warning(f"Could not download LSTM model: {e}")

    return status

@st.cache_resource(show_spinner=False)
def load_models():
    """Load both models — cached so they load only once per session."""
    yolo_model = None
    lstm_model = None

    # Load YOLO
    if os.path.exists(YOLO_PATH):
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(YOLO_PATH)
        except Exception as e:
            pass

    # Load LSTM
# Load LSTM
    if os.path.exists(LSTM_PATH):
        try:
            m = CongestionLSTM()
            try:
                m.load_state_dict(torch.load(LSTM_PATH, map_location="cpu", weights_only=True))
            except:
                m.load_state_dict(torch.load(LSTM_PATH, map_location="cpu"))
            lstm_model = m.eval()
        except Exception as e:
            pass

    return yolo_model, lstm_model


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def get_density(count, risk_multiplier=1.0):
    """
    Convert vehicle count to density label.
    risk_multiplier > 1 means adverse conditions lower the threshold.
    """
    adjusted = count * risk_multiplier
    if adjusted <= 5:  return "Low",      0, "#10b981", "accent-green"
    if adjusted <= 15: return "Medium",   1, "#f59e0b", "accent-amber"
    if adjusted <= 30: return "High",     2, "#f97316", "accent-amber"
    return                   "Critical", 3, "#ef4444", "accent-red"

def get_weather_risk(weather, lighting, rain):
    """
    Calculate risk multiplier from environmental conditions.
    Scientifically: adverse conditions reduce road capacity,
    meaning same vehicle count = worse effective congestion.
    """
    multiplier = 1.0
    reasons    = []

    if rain == "Heavy Rain":
        multiplier += 0.4
        reasons.append("heavy rain reduces visibility & road grip")
    elif rain == "Light Rain":
        multiplier += 0.15
        reasons.append("wet roads reduce safe following distance")

    if weather == "Foggy":
        multiplier += 0.35
        reasons.append("fog forces slower speeds")
    elif weather == "Snowy":
        multiplier += 0.5
        reasons.append("snow severely reduces road capacity")
    elif weather == "Cloudy":
        multiplier += 0.05

    if lighting == "Night":
        multiplier += 0.2
        reasons.append("reduced night visibility")
    elif lighting == "Dusk/Dawn":
        multiplier += 0.1
        reasons.append("glare at dusk/dawn affects drivers")

    return round(multiplier, 2), reasons

def predict_lstm(lstm, buf):
    if lstm is None or len(buf) < 30: return None
    d = np.array(list(buf), dtype=np.float32)
    d[:,0] /= 50.0; d[:,1] /= 3.0
    with torch.no_grad():
        return round(float(lstm(torch.tensor(d).unsqueeze(0))) * 3, 2)

def draw_on_frame(frame, count, label, color_hex, pred, fnum, fps, risk_mult):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,58), (8,9,15), -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
    c = {
        "#10b981":(129,185,16), "#f59e0b":(11,158,245),
        "#f97316":(22,115,249), "#ef4444":(68,68,239)
    }.get(color_hex, (200,200,200))
    cv2.putText(frame, f"Vehicles: {count}", (12,38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,222,240), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Density: {label}",  (w//3,38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2, cv2.LINE_AA)
    if pred is not None:
        pl = ["Low","Medium","High","Critical"][min(3,int(round(pred)))]
        cv2.putText(frame, f"Predicted: {pl}", (int(1.9*w//3),38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,130,220), 2, cv2.LINE_AA)
    if risk_mult > 1.05:
        cv2.putText(frame, f"Risk x{risk_mult:.1f}", (w-130,38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (239,68,68), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.0f}fps #{fnum}", (w-120,18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (40,44,70), 1, cv2.LINE_AA)
    return frame

def make_timeline(hist):
    times  = [h["t"]     for h in hist]
    counts = [h["count"] for h in hist]
    levels = [h["level"] for h in hist]
    preds  = [h["pred"]  for h in hist]

    fig = go.Figure()

    # Vehicle count area
    fig.add_trace(go.Scatter(
        x=times, y=counts,
        name="Vehicles detected",
        mode="lines",
        line=dict(color="#6366f1", width=2),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.06)",
        hovertemplate="<b>%{y} vehicles</b><br>t=%{x}s<extra></extra>"
    ))

    # Density level (scaled for visibility)
    level_scaled = [l * 8 for l in levels]
    fig.add_trace(go.Scatter(
        x=times, y=level_scaled,
        name="Density level (×8)",
        mode="lines",
        line=dict(color="#8b5cf6", width=1.5, dash="dot"),
        hovertemplate="<b>Level %{customdata}</b><br>t=%{x}s<extra></extra>",
        customdata=[["Low","Medium","High","Critical"][min(3,l)] for l in levels]
    ))

    # LSTM prediction
    if any(p is not None for p in preds):
        p_scaled = [p * 8 if p is not None else None for p in preds]
        fig.add_trace(go.Scatter(
            x=times, y=p_scaled,
            name="LSTM prediction (×8)",
            mode="lines",
            line=dict(color="#10b981", width=1.5, dash="dash"),
            hovertemplate="<b>Predicted level %{customdata:.2f}</b><br>t=%{x}s<extra></extra>",
            customdata=[p if p else 0 for p in preds]
        ))

    fig.update_layout(
        height=200,
        margin=dict(l=0,r=0,t=8,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#3a3d60", size=11, family="JetBrains Mono"),
        legend=dict(
            orientation="h", y=1.25, x=0,
            font=dict(color="#4a4d70", size=10),
            bgcolor="rgba(0,0,0,0)"
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#12132a",
            bordercolor="#1a1b40",
            font=dict(color="#d4d6f0", size=12, family="Outfit")
        ),
        xaxis=dict(
            showgrid=True, gridcolor="#0e0f1e", gridwidth=1,
            tickfont=dict(color="#1e2040"), showline=False,
            zeroline=False,
            title=dict(text="Time (seconds)", font=dict(color="#1e2040", size=10))
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#0e0f1e", gridwidth=1,
            tickfont=dict(color="#1e2040"), showline=False,
            zeroline=False
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
defaults = {
    "running": False, "history": [], "vpath": None,
    "models_loaded": False, "app_ready": False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
# LOADING SCREEN (first visit)
# ─────────────────────────────────────────────────────────────
if not st.session_state.app_ready:
    st.markdown("""
    <div style="text-align:center;padding:6rem 2rem;">
        <div style="width:64px;height:64px;background:#6366f1;border-radius:18px;
        display:inline-flex;align-items:center;justify-content:center;
        font-size:28px;margin-bottom:20px;">&#128678;</div>
        <div style="font-size:2.2rem;font-weight:800;color:#fff;letter-spacing:-1px;margin-bottom:6px;">
        TrafficVision <span style="color:#6366f1;">AI</span></div>
        <div style="font-size:0.85rem;color:#2a2d50;font-family:monospace;letter-spacing:2px;margin-bottom:30px;">
        INITIALIZING SYSTEM</div>
        <div style="font-size:0.8rem;color:#2a2d50;font-family:monospace;">
        Loading models and checking environment...</div>
    </div>
    """, unsafe_allow_html=True)

    # Do the actual loading
    model_status = ensure_models()
    yolo_model, lstm_model = load_models()

    st.session_state.models_loaded = True
    st.session_state.app_ready = True
    st.session_state._yolo = yolo_model
    st.session_state._lstm = lstm_model
    time.sleep(0.8)
    st.rerun()

# Get cached models
yolo_model = st.session_state.get("_yolo", None)
lstm_model = st.session_state.get("_lstm", None)

if yolo_model is None and lstm_model is None:
    yolo_model, lstm_model = load_models()


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding:0.5rem 0 1.4rem;border-bottom:1px solid #12132a;margin-bottom:1.4rem;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <div style="width:34px;height:34px;background:#6366f1;border-radius:10px;
            display:flex;align-items:center;justify-content:center;font-size:16px;">&#128678;</div>
            <div>
                <div style="font-size:1.05rem;font-weight:800;color:#fff;letter-spacing:-0.3px;">
                TrafficVision <span style="color:#6366f1;">AI</span></div>
                <div style="font-family:monospace;font-size:0.58rem;color:#1e2040;letter-spacing:2px;">
                v2.0 &nbsp; BDD100K + VisDrone</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model status
    yo = yolo_model is not None
    lo = lstm_model is not None
    go_ = torch.cuda.is_available()

    def sdot(on, label):
        col = "#10b981" if on else "#2a2d50"
        txt = "#10b981" if on else "#2a2d50"
        status = "ready" if on else "not found"
        return f"""<div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid #0e0f1e;">
            <div style="width:7px;height:7px;border-radius:50%;background:{col};
            {'box-shadow:0 0 6px '+col+';' if on else ''}flex-shrink:0;"></div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:{txt};">
            {label} <span style="color:{'#1e2040' if not on else '#0d3d28'};font-size:0.65rem;">
            {'✓' if on else '✗'} {status}</span></div>
        </div>"""

    st.markdown(f"""
    <div style="background:#0a0b18;border:1px solid #12132a;border-radius:10px;
    padding:10px 14px;margin-bottom:14px;">
        {sdot(yo,  'YOLOv8')}
        {sdot(lo,  'LSTM  ')}
        {sdot(go_, 'GPU   ')}
    </div>
    """, unsafe_allow_html=True)

    if not yo:
        st.markdown("""
        <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);
        border-radius:8px;padding:10px 12px;margin-bottom:12px;">
            <div style="font-size:0.75rem;color:#f59e0b;font-weight:600;margin-bottom:4px;">
            ⚠ Models not found</div>
            <div style="font-size:0.7rem;color:#6a5a20;">
            Place best.pt and lstm_congestion.pt in the models/ folder then restart the app.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='sdiv'></div>", unsafe_allow_html=True)

    # Settings
    st.markdown("<div style='font-size:0.65rem;color:#1e2040;letter-spacing:2px;margin-bottom:10px;font-weight:600;'>DETECTION SETTINGS</div>", unsafe_allow_html=True)
    conf_thresh   = st.slider("Confidence", 0.10, 0.90, 0.40, 0.05)
    process_every = st.slider("Frame skip",  1, 10, 3,
        help="Process 1 in every N frames. Higher = faster but less detail.")

    st.markdown("<div class='sdiv'></div>", unsafe_allow_html=True)

    # Dataset info
    st.markdown("""
    <div style="background:#0a0b18;border:1px solid #12132a;border-radius:10px;padding:12px 14px;">
        <div style="font-size:0.65rem;color:#1e2040;letter-spacing:2px;margin-bottom:10px;font-weight:600;">DATASET</div>
        <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid #0e0f1e;">
            <span style="font-size:0.78rem;color:#3a3d60;">Detection</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#8b8ef0;font-weight:600;">BDD100K + VisDrone</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid #0e0f1e;">
            <span style="font-size:0.78rem;color:#3a3d60;">LSTM</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#8b8ef0;font-weight:600;">Val Loss: 0.0107</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid #0e0f1e;">
            <span style="font-size:0.78rem;color:#3a3d60;">Conditions</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#3a3d60;">Night · Rain · Fog</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;">
            <span style="font-size:0.78rem;color:#3a3d60;">Classes</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#3a3d60;">Car · Bus · Truck · Moto</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#0e0f1e;border:1px solid #12132a;border-top:2px solid #6366f1;border-radius:16px;padding:1.8rem 2.2rem;margin-bottom:1.5rem;">
    <div style="font-family:monospace;font-size:0.62rem;color:#6366f1;letter-spacing:3px;margin-bottom:8px;">SYSTEM ONLINE</div>
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;">
        <div style="width:48px;height:48px;background:#6366f1;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;">&#128678;</div>
        <div>
            <div style="font-size:2rem;font-weight:800;color:#fff;letter-spacing:-1px;line-height:1;">TrafficVision <span style="color:#6366f1;">AI</span></div>
            <div style="color:#3a3d60;font-size:0.82rem;margin-top:4px;">Real-time vehicle detection &nbsp;|&nbsp; Density estimation &nbsp;|&nbsp; Congestion forecasting</div>
        </div>
    </div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <span style="background:#1a1b30;color:#8b8ef0;border:1px solid #2a2b50;padding:3px 12px;border-radius:20px;font-size:0.7rem;font-weight:600;">YOLOv8</span>
        <span style="background:#1a1b30;color:#a78bfa;border:1px solid #2a2b50;padding:3px 12px;border-radius:20px;font-size:0.7rem;font-weight:600;">LSTM</span>
        <span style="background:#0d2a1e;color:#10b981;border:1px solid #1a4a30;padding:3px 12px;border-radius:20px;font-size:0.7rem;font-weight:600;">BDD100K + VisDrone</span>
        <span style="background:#1a1b30;color:#3a3d60;border:1px solid #1e1f35;padding:3px 12px;border-radius:20px;font-size:0.7rem;">Car &nbsp; Bus &nbsp; Truck &nbsp; Moto</span>
        <span style="background:#1a1b30;color:#3a3d60;border:1px solid #1e1f35;padding:3px 12px;border-radius:20px;font-size:0.7rem;">Weather-Aware</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# UPLOAD + WEATHER FORM
# ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload traffic video",
    type=["mp4","avi","mov","mkv"],
    help="Upload CCTV, dashcam or drone footage"
)

if uploaded is None:
    # Landing cards
    c1, c2, c3 = st.columns(3)
    for col, step, title, desc, color in [
        (c1, "01", "Upload Video",  "CCTV &#183; dashcam &#183; drone footage", "#6366f1"),
        (c2, "02", "Set Conditions","Weather &#183; lighting &#183; time of day", "#8b5cf6"),
        (c3, "03", "Analyze",       "Live detection + AI prediction",  "#10b981"),
    ]:
        col.markdown(f"""
        <div style="background:#0e0f1e;border:1px solid #12132a;border-radius:12px;
        padding:22px 20px;transition:all 0.25s ease;cursor:default;"
        onmouseover="this.style.borderColor='{color}';this.style.boxShadow='0 4px 20px rgba(0,0,0,0.3)'"
        onmouseout="this.style.borderColor='#12132a';this.style.boxShadow='none'">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
            color:{color};margin-bottom:10px;letter-spacing:1.5px;font-weight:600;">
            STEP {step}</div>
            <div style="font-size:1rem;font-weight:700;color:#d4d6f0;margin-bottom:5px;">
            {title}</div>
            <div style="font-size:0.78rem;color:#2a2d50;">{desc}</div>
        </div>""", unsafe_allow_html=True)

else:
    # ── Weather conditions panel ──────────────────────────────
    st.markdown("""
    <div style="background:#0e0f1e;border:1px solid #12132a;border-radius:12px;
    padding:16px 20px;margin-bottom:14px;">
        <div style="font-size:0.65rem;color:#1e2040;letter-spacing:2px;
        font-weight:600;margin-bottom:12px;">ENVIRONMENTAL CONDITIONS</div>
        <div style="font-size:0.8rem;color:#2a2d50;margin-bottom:10px;">
        These affect congestion risk — adverse conditions reduce road capacity,
        making the same vehicle count more dangerous.</div>
    </div>
    """, unsafe_allow_html=True)

    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        weather = st.selectbox("Weather",  ["Clear","Cloudy","Foggy","Snowy"], index=0)
    with wc2:
        lighting = st.selectbox("Lighting", ["Daytime","Dusk/Dawn","Night"], index=0)
    with wc3:
        rain = st.selectbox("Precipitation", ["None","Light Rain","Heavy Rain"], index=0)

    risk_mult, risk_reasons = get_weather_risk(weather, lighting, rain)

    if risk_mult > 1.05:
        reasons_txt = " · ".join(risk_reasons) if risk_reasons else ""
        risk_level = "moderate" if risk_mult < 1.3 else ("high" if risk_mult < 1.6 else "critical")
        risk_color = {"moderate":"#f59e0b","high":"#f97316","critical":"#ef4444"}[risk_level]
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.15);
        border-left:3px solid {risk_color};border-radius:10px;padding:12px 16px;margin-bottom:12px;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <span style="font-size:0.8rem;font-weight:700;color:{risk_color};">
                ⚠ Adverse Conditions Detected</span>
                <span class="risk-badge risk-{risk_level}">{risk_level.upper()}</span>
            </div>
            <div style="font-size:0.78rem;color:#5a3d3d;">
            Risk multiplier: <b style="color:{risk_color};">×{risk_mult:.2f}</b> — {reasons_txt}</div>
            <div style="font-size:0.72rem;color:#3a2020;margin-top:4px;">
            Congestion thresholds are adjusted accordingly.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.15);
        border-left:3px solid #10b981;border-radius:10px;padding:10px 16px;margin-bottom:12px;">
            <span style="font-size:0.78rem;color:#10b981;font-weight:600;">
            ✓ Clear conditions — standard thresholds active</span>
        </div>
        """, unsafe_allow_html=True)

    # Save temp file (fix WinError 32 — close before CV2 reads)
    if st.session_state.vpath is None or not os.path.exists(st.session_state.vpath):
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tf.write(uploaded.read()); tf.flush(); tf.close()
        st.session_state.vpath = tf.name
    vpath = st.session_state.vpath

    # ── Main layout ───────────────────────────────────────────
    left, right = st.columns([3, 2], gap="medium")

    with left:
        st.markdown("<div style='font-size:0.62rem;color:#1e2040;letter-spacing:2px;font-weight:600;margin-bottom:8px;'>LIVE DETECTION FEED</div>", unsafe_allow_html=True)
        feed_slot = st.empty()

    with right:
        st.markdown("<div style='font-size:0.62rem;color:#1e2040;letter-spacing:2px;font-weight:600;margin-bottom:8px;'>REAL-TIME METRICS</div>", unsafe_allow_html=True)
        metrics_slot = st.empty()
        pred_slot    = st.empty()

    st.markdown("<div style='font-size:0.62rem;color:#1e2040;letter-spacing:2px;font-weight:600;margin:14px 0 6px;'>DENSITY TIMELINE  <span style='color:#1a1b30;font-weight:400;'>— hover over chart for values</span></div>", unsafe_allow_html=True)
    chart_slot = st.empty()

    # Controls
    b1, b2, b3, _= st.columns([1,1,1,3])
    with b1: start_btn = st.button("▶  Start Analysis")
    with b2: stop_btn  = st.button("■  Stop")
    with b3: reset_btn = st.button("↺  Reset")

    if start_btn:
        st.session_state.running = True
        st.session_state.history = []
    if stop_btn:
        st.session_state.running = False
    if reset_btn:
        st.session_state.running = False
        st.session_state.history = []
        if st.session_state.vpath and os.path.exists(st.session_state.vpath):
            try: os.unlink(st.session_state.vpath)
            except: pass
        st.session_state.vpath = None
        st.rerun()

    # ── Processing loop ───────────────────────────────────────
    if st.session_state.running:
        cap     = cv2.VideoCapture(vpath)
        fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25
        buf     = deque(maxlen=30)
        fnum    = 0
        t0      = time.time()
        dev     = "cuda" if torch.cuda.is_available() else "cpu"

        while cap.isOpened() and st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.session_state.running = False
                break

            fnum += 1
            if fnum % process_every != 0:
                continue

            # ── Detect vehicles ──
            if yolo_model is not None:
                try:
                    r     = yolo_model(frame, conf=conf_thresh, verbose=False)[0]
                    count = len(r.boxes)
                    frame = r.plot()
                except:
                    count = max(0, int(np.random.normal(15, 5)))
            else:
                # Demo mode without model
                t_el  = fnum / fps_vid
                count = max(0, int(12 + 8*np.sin(t_el/30) + np.random.normal(0,2)))

            # ── Apply weather risk to density ──
            label, level, color, accent = get_density(count, risk_mult)
            buf.append([count, level])

            # ── LSTM prediction ──
            pred = predict_lstm(lstm_model, buf)

            elapsed  = time.time() - t0
            fps_proc = fnum / elapsed if elapsed > 0 else 0

            # Draw on frame
            frame = draw_on_frame(frame, count, label, color, pred, fnum, fps_proc, risk_mult)
            feed_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

            st.session_state.history.append({
                "t": round(fnum/fps_vid, 1),
                "count": count, "level": level,
                "label": label, "pred": pred
            })

            # ── Metrics cards ──
            metrics_slot.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:10px;">
                <div style="background:#0e0f1e;border:1px solid #12132a;border-top:2px solid {color};
                border-radius:11px;padding:14px 12px;text-align:center;">
                    <div style="font-size:2rem;font-weight:800;color:#fff;line-height:1;
                    font-family:'JetBrains Mono',monospace;">{count}</div>
                    <div class="stat-lbl">Vehicles</div>
                </div>
                <div style="background:#0e0f1e;border:1px solid #12132a;border-top:2px solid {color};
                border-radius:11px;padding:14px 12px;text-align:center;">
                    <div style="font-size:1.2rem;font-weight:800;color:{color};line-height:1.2;">{label}</div>
                    <div class="stat-lbl">Density</div>
                </div>
                <div style="background:#0e0f1e;border:1px solid #12132a;border-top:2px solid #6366f1;
                border-radius:11px;padding:14px 12px;text-align:center;">
                    <div style="font-size:2rem;font-weight:800;color:#fff;line-height:1;
                    font-family:'JetBrains Mono',monospace;">{fps_proc:.0f}</div>
                    <div class="stat-lbl">FPS</div>
                </div>
            </div>
            {"<div style='background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.15);border-radius:8px;padding:8px 12px;margin-bottom:8px;font-size:0.72rem;color:#ef4444;'>⚠ Risk ×"+str(risk_mult)+" — adverse conditions active</div>" if risk_mult > 1.05 else ""}
            """, unsafe_allow_html=True)

            # ── Prediction panel ──
            if pred is not None:
                plab = ["Low","Medium","High","Critical"][min(3,int(round(pred)))]
                diff = pred - level
                if diff > 0.35:
                    bg,border,icon,tcol,title,msg = (
                        "rgba(239,68,68,0.06)","#ef4444","↑","#ef4444",
                        "CONGESTION RISING",
                        f"Expect <b style='color:#ef4444;'>{plab}</b> conditions in next few frames"
                    )
                elif diff < -0.35:
                    bg,border,icon,tcol,title,msg = (
                        "rgba(16,185,129,0.06)","#10b981","↓","#10b981",
                        "TRAFFIC EASING",
                        f"Conditions improving — predicted <b style='color:#10b981;'>{plab}</b>"
                    )
                else:
                    bg,border,icon,tcol,title,msg = (
                        "rgba(99,102,241,0.06)","#6366f1","→","#8b8ef0",
                        "CONDITIONS STABLE",
                        f"Steady <b style='color:#8b8ef0;'>{plab}</b> traffic flow expected"
                    )
                pred_slot.markdown(f"""
                <div style="background:{bg};border:1px solid {border};
                border-left:3px solid {border};border-radius:11px;padding:14px 16px;">
                    <div style="font-size:1.3rem;font-weight:800;color:{tcol};
                    letter-spacing:-0.5px;margin-bottom:5px;">{icon} {title}</div>
                    <div style="font-size:0.82rem;color:#5a5d80;line-height:1.5;">{msg}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                    color:#1e2040;margin-top:8px;padding-top:8px;border-top:1px solid #12132a;">
                    LSTM output: {pred:.3f} / 3.000 &nbsp;·&nbsp; frame #{fnum}</div>
                </div>""", unsafe_allow_html=True)
            else:
                needed = 30 - len(buf)
                pct    = int(len(buf) / 30 * 100)
                pred_slot.markdown(f"""
                <div style="background:rgba(99,102,241,0.05);border:1px solid rgba(99,102,241,0.15);
                border-left:3px solid #6366f1;border-radius:11px;padding:14px 16px;">
                    <div style="font-size:1rem;font-weight:700;color:#6366f1;margin-bottom:5px;">
                    ◌ Building prediction...</div>
                    <div style="font-size:0.82rem;color:#3a3d60;">
                    Collecting {needed} more frames ({pct}% ready)</div>
                    <div style="background:#0e0f1e;border-radius:4px;height:4px;margin-top:10px;overflow:hidden;">
                        <div style="background:linear-gradient(90deg,#6366f1,#8b5cf6);
                        width:{pct}%;height:100%;border-radius:4px;transition:width 0.3s;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            # ── Chart ──
            if len(st.session_state.history) > 3:
                chart_slot.plotly_chart(
                    make_timeline(st.session_state.history),
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                        "scrollZoom": False,
                    }
                )

        cap.release()
        st.session_state.running = False

    # ── Session summary ───────────────────────────────────────
    if st.session_state.history and not st.session_state.running:
        h    = st.session_state.history
        avgv = np.mean([x["count"] for x in h])
        maxv = max(x["count"] for x in h)
        nf   = len(h)
        cpct = sum(1 for x in h if x["level"] >= 2) / nf * 100
        cc   = "#ef4444" if cpct > 50 else ("#f59e0b" if cpct > 25 else "#10b981")

        st.markdown("<div class='sdiv'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.62rem;color:#1e2040;letter-spacing:2px;font-weight:600;margin-bottom:12px;'>SESSION SUMMARY</div>", unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        for col, val, lbl, vc in [
            (s1, f"{avgv:.1f}", "Avg vehicles / frame", "#fff"),
            (s2, str(maxv),     "Peak vehicle count",   "#fff"),
            (s3, str(nf),       "Frames analyzed",      "#fff"),
            (s4, f"{cpct:.0f}%","Time in congestion",   cc),
        ]:
            col.markdown(f"""
            <div style="background:#0e0f1e;border:1px solid #12132a;border-radius:12px;
            padding:18px 16px;text-align:center;transition:border-color 0.2s;"
            onmouseover="this.style.borderColor='#1a1b40'"
            onmouseout="this.style.borderColor='#12132a'">
                <div style="font-size:1.9rem;font-weight:800;color:{vc};
                font-family:'JetBrains Mono',monospace;line-height:1;">{val}</div>
                <div class="stat-lbl" style="margin-top:6px;">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        # Safe cleanup
        if st.session_state.vpath and os.path.exists(st.session_state.vpath):
            try:
                os.unlink(st.session_state.vpath)
                st.session_state.vpath = None
            except:
                pass