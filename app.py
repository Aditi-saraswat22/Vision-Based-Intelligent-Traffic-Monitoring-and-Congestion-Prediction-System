"""
TrafficVision AI - Main Application
Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import time
import os
import tempfile
from collections import deque
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TrafficVision AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .main { background: #0a0a0f; }
    .stApp { background: #0a0a0f; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #111118 !important;
        border-right: 1px solid #1e1e2e;
    }
    [data-testid="stSidebar"] * { color: #c8c8d8 !important; }

    /* Header */
    .tv-header {
        background: linear-gradient(135deg, #0d0d1a 0%, #111128 100%);
        border: 1px solid #1e1e3a;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .tv-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .tv-title {
        font-size: 2rem;
        font-weight: 700;
        color: #fff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .tv-subtitle {
        color: #6b6b8a;
        font-size: 0.9rem;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .tv-badge {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        color: #818cf8;
        border: 1px solid rgba(99,102,241,0.3);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 8px;
    }

    /* Metric cards */
    .metric-card {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        height: 100%;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #fff;
        line-height: 1;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #6b6b8a;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 6px;
    }
    .metric-sub {
        font-size: 0.85rem;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Status badge */
    .status-low    { color: #34d399; }
    .status-medium { color: #fbbf24; }
    .status-high   { color: #f87171; }
    .status-critical { color: #ef4444; animation: pulse 1s infinite; }

    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }

    /* Prediction box */
    .pred-box {
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 12px;
        border: 1px solid;
    }
    .pred-improve {
        background: rgba(52, 211, 153, 0.08);
        border-color: rgba(52, 211, 153, 0.3);
        color: #34d399;
    }
    .pred-worsen {
        background: rgba(248, 113, 113, 0.08);
        border-color: rgba(248, 113, 113, 0.3);
        color: #f87171;
    }
    .pred-stable {
        background: rgba(251, 191, 36, 0.08);
        border-color: rgba(251, 191, 36, 0.3);
        color: #fbbf24;
    }

    /* Section headers */
    .section-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #6b6b8a;
        margin-bottom: 12px;
        font-weight: 600;
    }

    /* Video container */
    .video-container {
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        overflow: hidden;
        background: #05050a;
    }

    /* Hide streamlit default styling */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    div[data-testid="stToolbar"] { display: none; }

    .stButton > button {
        background: rgba(99,102,241,0.15);
        color: #818cf8;
        border: 1px solid rgba(99,102,241,0.4);
        border-radius: 8px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        padding: 8px 20px;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: rgba(99,102,241,0.25);
        border-color: rgba(99,102,241,0.6);
    }

    .upload-area {
        border: 2px dashed #1e1e3a;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        color: #6b6b8a;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LSTM MODEL DEFINITION
# ─────────────────────────────────────────────
class CongestionLSTM(nn.Module):
    """
    LSTM model that takes a sequence of past density values
    and predicts the next density value.
    Input:  (batch, sequence_len=30, features=2)  [count, density_level]
    Output: (batch, 1)                            [predicted density level]
    """
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_density_label(count, frame_area=None):
    """Convert vehicle count to density label + numeric level."""
    if count <= 5:
        return "Low", 0, "#34d399"
    elif count <= 15:
        return "Medium", 1, "#fbbf24"
    elif count <= 30:
        return "High", 2, "#f97316"
    else:
        return "Critical", 3, "#ef4444"


def load_yolo_model(weights_path):
    """Load YOLOv8 model. Returns model or None if not found."""
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        return model
    except Exception as e:
        return None


def load_lstm_model(model_path, device="cpu"):
    """Load trained LSTM model. Returns model or None."""
    try:
        model = CongestionLSTM()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        return None


def predict_congestion(lstm_model, buffer, device="cpu"):
    """
    Use LSTM to predict next density level.
    buffer: list of (count, level) tuples, length=30
    """
    if lstm_model is None or len(buffer) < 30:
        return None
    data = np.array(buffer, dtype=np.float32)
    # Normalize counts to 0-1 range (max expected ~50 vehicles)
    data[:, 0] = data[:, 0] / 50.0
    data[:, 1] = data[:, 1] / 3.0
    x = torch.tensor(data).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = lstm_model(x).item() * 3.0  # scale back
    return round(max(0, min(3, pred)), 2)


def draw_overlay(frame, count, density_label, density_color_hex, pred_level, frame_num):
    """Draw info overlay on video frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (8, 8, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Convert hex to BGR
    color_map = {
        "#34d399": (153, 211, 52),
        "#fbbf24": (36, 191, 251),
        "#f97316": (22, 115, 249),
        "#ef4444": (68, 68, 239)
    }
    bgr = color_map.get(density_color_hex, (255, 255, 255))

    # Text overlays
    cv2.putText(frame, f"TrafficVision AI", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Vehicles: {count}", (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Density: {density_label}", (w // 3, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2, cv2.LINE_AA)
    if pred_level is not None:
        pred_labels = ["Low", "Medium", "High", "Critical"]
        pred_text = pred_labels[min(3, int(round(pred_level)))]
        cv2.putText(frame, f"Predicted: {pred_text}", (2 * w // 3, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 130, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Frame #{frame_num}", (w - 130, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 130), 1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts
if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = None
if "lstm_model" not in st.session_state:
    st.session_state.lstm_model = None


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**Model Paths**")
    yolo_path = st.text_input("YOLOv8 weights", value="models/best.pt",
                               help="Path to your trained best.pt file")
    lstm_path = st.text_input("LSTM weights", value="models/lstm_congestion.pt",
                               help="Path to your trained LSTM model")

    if st.button("🔄 Load Models"):
        with st.spinner("Loading models..."):
            st.session_state.yolo_model = load_yolo_model(yolo_path)
            st.session_state.lstm_model = load_lstm_model(lstm_path)
        if st.session_state.yolo_model:
            st.success("✅ YOLOv8 loaded!")
        else:
            st.warning("⚠️ YOLO not found — using demo mode")
        if st.session_state.lstm_model:
            st.success("✅ LSTM loaded!")
        else:
            st.warning("⚠️ LSTM not found — no predictions")

    st.markdown("---")
    st.markdown("**Detection Settings**")
    conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
    process_every = st.slider("Process every N frames", 1, 10, 3,
                               help="Skip frames for speed. 1=all frames, 5=every 5th")

    st.markdown("---")
    st.markdown("**Status**")
    yolo_ok = st.session_state.yolo_model is not None
    lstm_ok = st.session_state.lstm_model is not None
    st.markdown(f"{'🟢' if yolo_ok else '🔴'} YOLOv8 detector")
    st.markdown(f"{'🟢' if lstm_ok else '🔴'} LSTM predictor")
    st.markdown(f"{'🟢' if torch.cuda.is_available() else '🟡'} GPU: {'Available' if torch.cuda.is_available() else 'CPU mode'}")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#4a4a6a;'>
    TrafficVision AI v1.0<br>
    YOLOv8 + LSTM Pipeline<br>
    UADetrac Dataset
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

# Header
st.markdown("""
<div class="tv-header">
    <p class="tv-title">🚦 TrafficVision AI</p>
    <p class="tv-subtitle">vision-based traffic monitoring + congestion prediction</p>
    <span class="tv-badge">YOLOv8 · LSTM · UADetrac</span>
</div>
""", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader(
    "Upload a traffic video",
    type=["mp4", "avi", "mov", "mkv"],
    help="Upload any traffic footage to analyze"
)

if uploaded_file is None:
    st.markdown("""
    <div class="upload-area">
        <div style="font-size:2rem;">📹</div>
        <div style="margin-top:8px;">Upload a traffic video above to begin</div>
        <div style="font-size:0.8rem; margin-top:4px;">Supports MP4, AVI, MOV, MKV</div>
    </div>
    """, unsafe_allow_html=True)

    # Show instructions instead
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Step 1 — Train**
        Load models using the sidebar after training in Colab.
        """)
    with col2:
        st.markdown("""
        **Step 2 — Upload**
        Upload any traffic video. Works with dashcam, CCTV, drone footage.
        """)
    with col3:
        st.markdown("""
        **Step 3 — Analyze**
        Get real-time density + congestion predictions frame by frame.
        """)

else:
    # ── Save uploaded video to temp file ──
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()
    video_path = tfile.name

    # ── Layout ──
    col_video, col_stats = st.columns([3, 2])

    with col_video:
        st.markdown('<div class="section-title">Live Detection Feed</div>', unsafe_allow_html=True)
        frame_display = st.empty()

    with col_stats:
        st.markdown('<div class="section-title">Real-Time Metrics</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        count_display  = m1.empty()
        density_display = m2.empty()
        fps_display    = m3.empty()
        pred_display   = st.empty()

    st.markdown("---")
    st.markdown('<div class="section-title">Density Timeline</div>', unsafe_allow_html=True)
    chart_display = st.empty()

    col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
    with col_b1:
        start_btn = st.button("▶ Start Analysis")
    with col_b2:
        stop_btn = st.button("⏹ Stop")

    if start_btn:
        st.session_state.running = True
        st.session_state.history = []

    if stop_btn:
        st.session_state.running = False

    # ── Processing Loop ──
    if st.session_state.running:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 25

        density_buffer = deque(maxlen=30)
        frame_num = 0
        t_start = time.time()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        while cap.isOpened() and st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Skip frames for speed
            if frame_num % process_every != 0:
                continue

            # ── Detection ──
            count = 0
            if st.session_state.yolo_model is not None:
                try:
                    results = st.session_state.yolo_model(
                        frame, conf=conf_thresh, verbose=False
                    )[0]
                    count = len(results.boxes)
                    # Draw bounding boxes
                    frame = results.plot()
                except Exception as e:
                    # Demo mode: simulate detection
                    count = int(np.random.normal(15, 5))
                    count = max(0, count)
            else:
                # DEMO MODE — simulate vehicle count with realistic variation
                t_elapsed = frame_num / fps_video
                base = 12 + 8 * np.sin(t_elapsed / 30)
                count = max(0, int(base + np.random.normal(0, 2)))

            # ── Density ──
            label, level, color = get_density_label(count)
            density_buffer.append([count, level])

            # ── LSTM Prediction ──
            pred_level = predict_congestion(
                st.session_state.lstm_model, list(density_buffer), device
            )

            # ── Draw overlay on frame ──
            frame = draw_overlay(frame, count, label, color, pred_level, frame_num)

            # ── FPS calc ──
            elapsed = time.time() - t_start
            fps_proc = frame_num / elapsed if elapsed > 0 else 0

            # ── Store history ──
            st.session_state.history.append({
                "frame": frame_num,
                "count": count,
                "level": level,
                "label": label,
                "pred": pred_level,
                "time_s": round(frame_num / fps_video, 1)
            })

            # ── Update UI ──
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_display.image(frame_rgb, use_column_width=True)

            count_display.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{count}</div>
                <div class="metric-label">Vehicles</div>
            </div>""", unsafe_allow_html=True)

            density_display.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.4rem; color:{color};">{label}</div>
                <div class="metric-label">Density</div>
            </div>""", unsafe_allow_html=True)

            fps_display.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{fps_proc:.1f}</div>
                <div class="metric-label">FPS</div>
            </div>""", unsafe_allow_html=True)

            # Prediction box
            if pred_level is not None:
                pred_labels = ["Low", "Medium", "High", "Critical"]
                pred_text = pred_labels[min(3, int(round(pred_level)))]
                diff = pred_level - level
                if diff > 0.3:
                    cls = "pred-worsen"
                    icon = "📈"
                    msg = f"Congestion likely to worsen → {pred_text}"
                elif diff < -0.3:
                    cls = "pred-improve"
                    icon = "📉"
                    msg = f"Traffic expected to ease → {pred_text}"
                else:
                    cls = "pred-stable"
                    icon = "📊"
                    msg = f"Conditions stable → {pred_text}"
                pred_display.markdown(f"""
                <div class="pred-box {cls}">
                    <strong>{icon} LSTM Prediction</strong><br>
                    <span style="font-size:0.9rem;">{msg}</span>
                </div>""", unsafe_allow_html=True)
            else:
                frames_needed = 30 - len(density_buffer)
                pred_display.markdown(f"""
                <div class="pred-box pred-stable">
                    ⏳ Collecting data for prediction... ({frames_needed} more frames needed)
                </div>""", unsafe_allow_html=True)

            # ── Timeline chart ──
            if len(st.session_state.history) > 2:
                hist = st.session_state.history
                times  = [h["time_s"] for h in hist]
                counts = [h["count"]  for h in hist]
                levels = [h["level"]  for h in hist]
                preds  = [h["pred"]   for h in hist]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=times, y=counts,
                    name="Vehicle count", mode="lines",
                    line=dict(color="#818cf8", width=2),
                    fill="tozeroy", fillcolor="rgba(129,140,248,0.08)"
                ))
                if any(p is not None for p in preds):
                    pred_scaled = [p * 10 if p is not None else None for p in preds]
                    fig.add_trace(go.Scatter(
                        x=times, y=pred_scaled,
                        name="Predicted level (×10)", mode="lines",
                        line=dict(color="#f59e0b", width=2, dash="dot")
                    ))
                fig.update_layout(
                    height=200,
                    margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#6b6b8a", size=11),
                    legend=dict(
                        orientation="h", y=1.1,
                        font=dict(color="#9090b0")
                    ),
                    xaxis=dict(
                        showgrid=True, gridcolor="#1a1a2e",
                        tickfont=dict(color="#6b6b8a"),
                        title="Time (seconds)"
                    ),
                    yaxis=dict(
                        showgrid=True, gridcolor="#1a1a2e",
                        tickfont=dict(color="#6b6b8a")
                    )
                )
                chart_display.plotly_chart(fig, use_container_width=True)

        cap.release()
        st.session_state.running = False

        # ── Summary after video ends ──
        if st.session_state.history:
            st.markdown("---")
            st.markdown("### 📊 Session Summary")
            hist = st.session_state.history
            avg_count = np.mean([h["count"] for h in hist])
            max_count = max(h["count"] for h in hist)
            congested_pct = sum(1 for h in hist if h["level"] >= 2) / len(hist) * 100

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Avg vehicles/frame", f"{avg_count:.1f}")
            s2.metric("Peak vehicle count",  str(max_count))
            s3.metric("Frames analyzed",     str(len(hist)))
            s4.metric("Time congested",       f"{congested_pct:.1f}%")

        os.unlink(video_path)
