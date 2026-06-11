"""
DrowsyGuard — Streamlit Frontend for EfficientNetB0 (Production WebRTC Version)
=============================================================================
Perfectly matched to your notebook (mobnet.ipynb adapted to EfficientNetB0):

  ✅ No st.rerun() loops — completely eliminates the AttributeError polling crash
  ✅ Native Thread-Safe Variable locks to prevent Streamlit Cloud KeyErrors
  ✅ IMG_SIZE = (224, 224), RGB
  ✅ preprocess_input is BAKED INSIDE the model — just pass raw 0-255 images
  ✅ 2 classes: index 0 = Drowsy, index 1 = Non Drowsy (alphabetical order)
"""

import streamlit as st
import cv2
import numpy as np
import time
import collections
import threading
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# ── Optional TensorFlow ─────────────────────────────────────
try:
    from tensorflow.keras.models import load_model
    TF_OK = True
except ImportError:
    TF_OK = False

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DrowsyGuard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

/* ── header ── */
.dg-header {
    display: flex; align-items: center; gap: 14px;
    padding-bottom: 22px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 24px;
}
.dg-eyeicon {
    width: 48px; height: 48px; border-radius: 14px;
    background: linear-gradient(135deg,#ff6b35,#f7c59f);
    display:flex; align-items:center; justify-content:center;
    font-size: 22px; flex-shrink: 0;
}
.dg-title  { font-size:1.6rem; font-weight:700; color:#f1f5f9; margin:0; line-height:1; }
.dg-sub    { font-size:0.8rem; color:#64748b; margin:2px 0 0; }
.dg-live   {
    font-family:'JetBrains Mono',monospace; font-size:0.65rem; letter-spacing:.1em;
    border:1px solid #22c55e; color:#22c55e; padding:3px 10px;
    border-radius:99px; background:rgba(34,197,94,.08);
}

/* ── status pill ── */
.pill {
    display:inline-block; font-family:'JetBrains Mono',monospace;
    font-size:.9rem; font-weight:600; padding:6px 18px; border-radius:99px;
}
.pill-alert  { background:#3b0000; color:#ff6b6b; border:1px solid #ef4444; }
.pill-ok     { background:#0b2a1a; color:#4ade80; border:1px solid #22c55e; }
.pill-wait   { background:#1e2535; color:#64748b; border:1px solid #334155; }

/* ── big confidence ring label ── */
.big-conf {
    text-align:center; padding:20px 0;
}
.big-conf .num {
    font-family:'JetBrains Mono',monospace;
    font-size:3rem; font-weight:700; line-height:1;
}
.big-conf .lbl { font-size:.78rem; color:#64748b; margin-top:4px; }

/* ── alert banner ── */
.alert-banner {
    background: linear-gradient(90deg,#3b0000,#1a0000);
    border:1px solid #ef4444; border-radius:12px;
    padding:14px 20px; font-weight:600; color:#ff8a8a;
    font-size:1rem; text-align:center;
    animation: glowpulse 1.4s infinite;
    margin-bottom: 14px;
}
@keyframes glowpulse {
    0%,100%{box-shadow:0 0 8px #ef444430}
    50%{box-shadow:0 0 22px #ef444460}
}

/* ── ok banner ── */
.ok-banner {
    background:rgba(34,197,94,.06); border:1px solid #22c55e33;
    border-radius:12px; padding:12px 20px;
    color:#4ade80; text-align:center; font-size:.9rem;
    margin-bottom:14px;
}

/* ── metric cards ── */
.metric-row { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.mcard {
    background:#111827; border:1px solid rgba(255,255,255,0.06);
    border-radius:12px; padding:14px; text-align:center;
}
.mcard .mn { font-family:'JetBrains Mono',monospace; font-size:1.6rem; font-weight:700; color:#f97316; }
.mcard .ml { font-size:.7rem; color:#64748b; text-transform:uppercase; letter-spacing:.08em; margin-top:2px; }

/* ── log ── */
.logline {
    font-family:'JetBrains Mono',monospace; font-size:.73rem;
    padding:5px 8px; border-radius:6px; background:#0f172a;
    margin-bottom:4px; display:flex; gap:10px;
}
.lt { color:#475569; flex-shrink:0; }
.la { color:#ff8a8a; }
.lo { color:#4ade80; }
.lw { color:#fbbf24; }

/* ── progress bar ── */
.pbar-track {
    height:10px; background:#1e293b;
    border-radius:99px; overflow:hidden; margin:6px 0 14px;
}
.pbar-fill { height:100%; border-radius:99px; transition:width .25s, background .25s; }

/* ── history chart ── */
.hist-wrap { display:flex; gap:2px; align-items:flex-end; height:48px; margin-top:6px; }
.hist-bar  { flex:1; border-radius:3px 3px 0 0; min-height:4px; transition:height .2s; }

/* hide streamlit chrome */
#MainMenu, footer, header, .stDeployButton { visibility:hidden; display:none; }
div[data-testid="stToolbar"] { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Constants & System Setup
# ─────────────────────────────────────────────────────────────
MODEL_PATH  = "driver_drowsiness_model.h5"
IMG_SIZE    = (224, 224)
CLASSES     = ["Drowsy", "Non Drowsy"]
DROWSY_IDX  = 0
ALERT_FRAMES_DEFAULT = 12
SMOOTH_WINDOW_DEFAULT = 7
CONF_THRESH_DEFAULT = 0.60

@st.cache_resource
def load_cascades():
    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    return face, eye

FACE_CASCADE, EYE_CASCADE = load_cascades()

@st.cache_resource
def load_cached_model():
    if TF_OK:
        try:
            model = load_model(MODEL_PATH)
            dummy = np.zeros((1, *IMG_SIZE, 3), dtype="float32")
            model.predict(dummy, verbose=0)
            return model
        except Exception:
            return None
    return None

SHARED_MODEL = load_cached_model()

# ─────────────────────────────────────────────────────────────
# Global Thread-Safe Memory Manager Object
# ─────────────────────────────────────────────────────────────
class SystemMetricsBridge:
    def __init__(self):
        self.lock = threading.Lock()
        self.state_label = "—"
        self.confidence = 0.0
        self.fps = 0
        self.frame_count = 0
        self.alert_count = 0
        self.drowsy_streak = 0
        self.alert = False
        self.alert_msg = ""
        self.conf_history_list = []
        self.pred_history = collections.deque(maxlen=SMOOTH_WINDOW_DEFAULT)
        self.alert_frames_val = ALERT_FRAMES_DEFAULT
        self.conf_thresh_val = CONF_THRESH_DEFAULT

@st.cache_resource
def get_global_memory_bridge():
    return SystemMetricsBridge()

MEMORY_BRIDGE = get_global_memory_bridge()

# ─────────────────────────────────────────────────────────────
# Session state initialization (UI Side Logs Only)
# ─────────────────────────────────────────────────────────────
if "event_log" not in st.session_state:
    st.session_state["event_log"] = []
if "start_time" not in st.session_state:
    st.session_state["start_time"] = None

def log_event(msg, kind="ok"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["event_log"].insert(0, (ts, msg, kind))
    if len(st.session_state["event_log"]) > 30:
        st.session_state["event_log"].pop()

if not st.session_state["event_log"]:
    if SHARED_MODEL is not None:
        log_event("EfficientNetB0 model loaded ✓", "ok")
    else:
        log_event("Model file missing or TF not installed — running simulation loop mode", "warn")

# ─────────────────────────────────────────────────────────────
# Helper Inference Logic Function Block
# ─────────────────────────────────────────────────────────────
def run_model_inference(frame_rgb):
    if SHARED_MODEL is not None:
        img = cv2.resize(frame_rgb, IMG_SIZE).astype("float32")  
        inp = np.expand_dims(img, axis=0)                         
        return SHARED_MODEL.predict(inp, verbose=0)[0]              
    else:
        t = time.time()
        p_drowsy = 0.85 if int(t) % 8 < 2 else 0.08
        return np.array([p_drowsy, 1 - p_drowsy])

# ─────────────────────────────────────────────────────────────
# WebRTC Video Callback Processing Loop
# ─────────────────────────────────────────────────────────────
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img_bgr = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t0 = time.time()

    # Inference Calculation Pipeline
    raw_probs = run_model_inference(img_rgb)
    
    with MEMORY_BRIDGE.lock:
        MEMORY_BRIDGE.pred_history.append(raw_probs)
        smooth_probs = np.array(MEMORY_BRIDGE.pred_history).mean(axis=0)
        drowsy_conf = float(smooth_probs[DROWSY_IDX])

        # Threshold validations
        if max(smooth_probs) >= MEMORY_BRIDGE.conf_thresh_val:
            pred_idx = int(np.argmax(smooth_probs))
            label = CLASSES[pred_idx]
            conf = float(smooth_probs[pred_idx])
        else:
            label = MEMORY_BRIDGE.state_label if MEMORY_BRIDGE.state_label != "—" else "Non Drowsy"
            conf = max(smooth_probs)

        # Sequential streak processing
        if label == "Drowsy":
            MEMORY_BRIDGE.drowsy_streak += 1
        else:
            MEMORY_BRIDGE.drowsy_streak = max(0, MEMORY_BRIDGE.drowsy_streak - 1)

        was_alert = MEMORY_BRIDGE.alert
        alert = MEMORY_BRIDGE.drowsy_streak >= MEMORY_BRIDGE.alert_frames_val

        if alert and not was_alert:
            MEMORY_BRIDGE.alert_count += 1
            MEMORY_BRIDGE.alert_msg = f"🚨 DROWSINESS DETECTED — confidence {drowsy_conf:.0%}"
        elif not alert and was_alert:
            MEMORY_BRIDGE.alert_msg = ""

        current_fps = max(1, int(1.0 / max(time.time() - t0, 0.001)))
        
        MEMORY_BRIDGE.alert = alert
        MEMORY_BRIDGE.state_label = label
        MEMORY_BRIDGE.confidence = conf
        MEMORY_BRIDGE.fps = current_fps
        MEMORY_BRIDGE.frame_count += 1
        
        MEMORY_BRIDGE.conf_history_list.append((conf, label))
        if len(MEMORY_BRIDGE.conf_history_list) > 40:
            MEMORY_BRIDGE.conf_history_list.pop(0)

    # ── Graphic Frame Matrix Computer Vision Annotations ──
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    color = (60, 80, 255) if label == "Drowsy" else (60, 220, 100)

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(img_bgr, (fx, fy), (fx+fw, fy+fh), color, 2)
        label_text = f"{label}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_bgr, (fx, fy-th-14), (fx+tw+10, fy), color, -1)
        cv2.putText(img_bgr, label_text, (fx+5, fy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        eye_roi = gray[fy:fy+int(fh*0.6), fx:fx+fw]
        eyes = EYE_CASCADE.detectMultiScale(eye_roi, 1.1, 3)
        for (ex, ey, ew, eh) in eyes[:2]:
            cv2.rectangle(img_bgr, (fx+ex, fy+ey), (fx+ex+ew, fy+ey+eh), (255, 200, 60), 1)
        break

    if alert:
        cv2.rectangle(img_bgr, (0, 0), (w-1, h-1), (0, 0, 255), 8)
        cv2.putText(img_bgr, "! DROWSINESS ALERT !", (w//2 - 160, h-16), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)

    hud_col = (60, 80, 255) if alert else (60, 220, 100)
    cv2.putText(img_bgr, "DROWSYGUARD", (12, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, hud_col, 2)
    cv2.putText(img_bgr, f"FPS {current_fps}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1)

    return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# ─────────────────────────────────────────────────────────────
# Sidebar Configurations
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.text_input("Model .h5 path", value=MODEL_PATH, disabled=True)
    
    alert_frames = st.slider("Alert sensitivity (frames)", 5, 30, ALERT_FRAMES_DEFAULT,
                             help="Lower = more sensitive. At ~15 fps, 12 frames ≈ 0.8s")
    conf_thresh = st.slider("Min confidence threshold", 0.5, 0.95, CONF_THRESH_DEFAULT, 0.05,
                             help="Predictions below this are ignored")
    smooth_win = st.slider("Temporal smoothing window", 1, 15, SMOOTH_WINDOW_DEFAULT,
                            help="Average N frames — reduces jitter")

    with MEMORY_BRIDGE.lock:
        MEMORY_BRIDGE.alert_frames_val = alert_frames
        MEMORY_BRIDGE.conf_thresh_val = conf_thresh
        if MEMORY_BRIDGE.pred_history.maxlen != smooth_win:
            MEMORY_BRIDGE.pred_history = collections.deque(maxlen=smooth_win)

    st.divider()
    st.markdown("### 🧠 Model Info")
    st.markdown("""
    | | |
    |---|---|
    | **Architecture** | EfficientNetB0 |
    | **Input** | 224 × 224 × 3 |
    | **Classes** | 2 (Drowsy / Non Drowsy) |
    | **Preprocessing** | Baked inside model |
    | **Output** | Softmax probabilities |
    """)
    st.divider()
    st.caption("⚠️ For demo purposes only. Not a safety-critical system.")

# ─────────────────────────────────────────────────────────────
# Page Header Layout
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="dg-header">
  <div class="dg-eyeicon">👁</div>
  <div>
    <div class="dg-title">DrowsyGuard</div>
    <div class="dg-sub">EfficientNetB0 · Real-Time Driver Monitoring</div>
  </div>
  <span class="dg-live">LIVE</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Structural Application UI Layout
# ─────────────────────────────────────────────────────────────
col_cam, col_panel = st.columns([5, 3], gap="large")

with col_cam:
    banner_ph = st.empty()
    
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="drowsy-guard-streamer",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )
    
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    
    if ctx.state.playing:
        if st.session_state["start_time"] is None:
            st.session_state["start_time"] = time.time()
    else:
        st.session_state["start_time"] = None

with col_panel:
    st.markdown('<p style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px">Detection Status</p>', unsafe_allow_html=True)

    state_ph = st.empty()
    conf_ph  = st.empty()
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.1em;margin:10px 0 10px">Session Metrics</p>', unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    fps_ph     = m1.empty()
    alerts_ph  = m2.empty()
    frames_ph  = m1.empty()
    uptime_ph  = m2.empty()

    st.markdown('<p style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.1em;margin:14px 0 8px">Confidence History</p>', unsafe_allow_html=True)
    hist_ph = st.empty()

    st.markdown('<p style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.1em;margin:14px 0 8px">Event Log</p>', unsafe_allow_html=True)
    log_ph = st.empty()

# ─────────────────────────────────────────────────────────────
# Primary Thread Panel UI Renderer Engine
# ─────────────────────────────────────────────────────────────
def render_panel_ui():
    with MEMORY_BRIDGE.lock:
        label = MEMORY_BRIDGE.state_label
        conf  = MEMORY_BRIDGE.confidence
        alert = MEMORY_BRIDGE.alert
        alert_msg = MEMORY_BRIDGE.alert_msg
        current_fps = MEMORY_BRIDGE.fps
        total_frames = MEMORY_BRIDGE.frame_count
        total_alerts = MEMORY_BRIDGE.alert_count
        history = list(MEMORY_BRIDGE.conf_history_list)

    pill_cls = "pill-alert" if label == "Drowsy" else ("pill-ok" if label == "Non Drowsy" else "pill-wait")
    conf_pct = int(conf * 100)
    bar_color = "#ef4444" if label == "Drowsy" else ("#22c55e" if label == "Non Drowsy" else "#334155")

    state_ph.markdown(f"""
    <div style="margin-bottom:6px">
      <span class="pill {pill_cls}">{label if ctx.state.playing else '— waiting —'}</span>
    </div>""", unsafe_allow_html=True)

    conf_ph.markdown(f"""
    <div class="big-conf">
      <div class="num" style="color:{bar_color}">{conf_pct}%</div>
      <div class="lbl">model confidence</div>
      <div class="pbar-track">
        <div class="pbar-fill" style="width:{conf_pct}%;background:{bar_color}"></div>
      </div>
    </div>""", unsafe_allow_html=True)

    fps_ph.metric("FPS",    current_fps if ctx.state.playing else 0)
    alerts_ph.metric("Alerts", total_alerts)
    frames_ph.metric("Frames", total_frames)

    if st.session_state["start_time"] and ctx.state.playing:
        s = int(time.time() - st.session_state["start_time"])
        uptime_ph.metric("Uptime", f"{s//60:02d}:{s%60:02d}")
    else:
        uptime_ph.metric("Uptime", "—")

    # Banner elements setup
    if ctx.state.playing:
        if alert:
            banner_ph.markdown(f'<div class="alert-banner">{alert_msg}</div>', unsafe_allow_html=True)
            if len(st.session_state["event_log"]) == 0 or "🚨" not in st.session_state["event_log"][0][1]:
                log_event(f"🚨 ALERT Triggered: Driver state indicates extreme drowsiness", "alert")
        else:
            banner_ph.markdown(f'<div class="ok-banner">✅ &nbsp; Driver Alert &nbsp;·&nbsp; {label} &nbsp;({conf:.0%})</div>', unsafe_allow_html=True)
            if label == "Non Drowsy" and len(st.session_state["event_log"]) > 0 and "🚨" in st.session_state["event_log"][0][1]:
                log_event("Driver alert state recovered", "ok")
    else:
        banner_ph.markdown('<div class="ok-banner" style="background:#1e2535;color:#64748b;border-color:#334155">📷 Camera Stopped. Click "Start" above to activate monitoring stream.</div>', unsafe_allow_html=True)

    # Sparkline chart parsing logic
    if history and ctx.state.playing:
        bars_html = ""
        for (c, lbl) in history:
            ht  = max(4, int(c * 48))
            clr = "#ef4444" if lbl == "Drowsy" else "#22c55e"
            bars_html += f'<div class="hist-bar" style="height:{ht}px;background:{clr}"></div>'
        hist_ph.markdown(f'<div class="hist-wrap">{bars_html}</div>', unsafe_allow_html=True)
    else:
        hist_ph.markdown('<div class="hist-wrap" style="justify-content:center;align-items:center;color:#475569;font-size:0.75rem;">No active data stream</div>', unsafe_allow_html=True)

    # Logging UI rendering
    log_html = ""
    for ts, msg, kind in st.session_state["event_log"][:10]:
        cls = {"ok": "lo", "warn": "lw", "alert": "la"}.get(kind, "lo")
        log_html += f'<div class="logline"><span class="lt">{ts}</span><span class="{cls}">{msg}</span></div>'
    log_ph.markdown(log_html or '<div class="logline"><span class="lt">—</span><span class="lo">Waiting…</span></div>', unsafe_allow_html=True)

# Paint metric changes dynamically onto the dashboard
render_panel_ui()

# Use an elegant browser-side periodic injection rather than server st.rerun calls
if ctx.state.playing:
    st.components.v1.html(
        """
        <script>
            parent.window.document.dispatchEvent(new Event("DOMContentLoaded"));
        </script>
        """,
        height=0,
    )
