"""
DrowsyGuard — Streamlit Frontend for EfficientNetB0 (WebRTC Version)
===================================================================
Perfectly matched to your notebook (mobnet.ipynb adapted to EfficientNetB0):

  ✅ Browser-based webcam streaming via streamlit-webrtc (Cloud Compatible)
  ✅ IMG_SIZE = (224, 224), RGB
  ✅ preprocess_input is BAKED INSIDE the model — just pass raw 0-255 images
  ✅ 2 classes: index 0 = Drowsy, index 1 = Non Drowsy  (alphabetical order)
  ✅ Dense(2, softmax) output
  ✅ Temporal smoothing for stable realtime predictions
  ✅ Rolling alert logic with cooldown to avoid false positives

Run:
    pip install streamlit opencv-python-headless tensorflow numpy streamlit-webrtc av
    streamlit run app.py

Place your model at:  model/efficientnet_drowsiness.h5
"""

import streamlit as st
import cv2
import numpy as np
import time
import collections
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
# Constants — matched EXACTLY to your notebook
# ─────────────────────────────────────────────────────────────
MODEL_PATH  = "model/efficientnet_drowsiness.h5"
IMG_SIZE    = (224, 224)          # your notebook Cell 2
CLASSES     = ["Drowsy", "Non Drowsy"]
DROWSY_IDX  = 0                   # 'Drowsy' is index 0 alphabetically
ALERT_FRAMES = 12                 # consecutive drowsy frames before alert
SMOOTH_WINDOW = 7                 # rolling average window for predictions
CONF_THRESH  = 0.60               # minimum confidence to trust prediction

# OpenCV face cascades
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# STUN configuration to support WebRTC connections across networks (essential for Cloud deployments)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

ss("model",          None)
ss("model_loaded",   False)
ss("alert",          False)
ss("alert_msg",      "")
ss("state_label",    "—")
ss("confidence",     0.0)
ss("fps",            0)
ss("alert_count",    0)
ss("frame_count",    0)
ss("start_time",     None)
ss("event_log",      [])
ss("pred_history",   collections.deque(maxlen=SMOOTH_WINDOW))
ss("drowsy_streak",  0)
ss("conf_history",   collections.deque(maxlen=40))

# ─────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────
def load_effnet():
    if st.session_state["model_loaded"]:
        return
    if TF_OK:
        try:
            st.session_state["model"] = load_model(MODEL_PATH)
            # Warm up the model with a dummy pass
            dummy = np.zeros((1, *IMG_SIZE, 3), dtype="float32")
            st.session_state["model"].predict(dummy, verbose=0)
            st.session_state["model_loaded"] = True
            log_event("EfficientNetB0 model loaded ✓", "ok")
        except Exception as e:
            log_event(f"Model load failed: {e} — demo mode", "warn")
            st.session_state["model_loaded"] = True
    else:
        log_event("TensorFlow not installed — demo mode active", "warn")
        st.session_state["model_loaded"] = True

def log_event(msg, kind="ok"):
    ts = datetime.now().strftime("%H:%M:%S")
    if "event_log" not in st.session_state:
        st.session_state["event_log"] = []
    st.session_state["event_log"].insert(0, (ts, msg, kind))
    if len(st.session_state["event_log"]) > 30:
        st.session_state["event_log"].pop()

# Load model instantly on setup script execution
load_effnet()

# ─────────────────────────────────────────────────────────────
# Inference & Logic Framework
# ─────────────────────────────────────────────────────────────
def predict_frame(frame_rgb):
    model = st.session_state["model"]

    img = cv2.resize(frame_rgb, IMG_SIZE).astype("float32")  # 0-255 float32
    inp = np.expand_dims(img, axis=0)                         # (1,224,224,3)

    if model is not None:
        probs = model.predict(inp, verbose=0)[0]              # [p_drowsy, p_awake]
    else:
        # Demo context: simulate cyclical state tracking
        t = time.time()
        p_drowsy = 0.85 if int(t) % 8 < 2 else 0.08
        probs = np.array([p_drowsy, 1 - p_drowsy])

    return probs

def smooth_prediction(probs):
    st.session_state["pred_history"].append(probs)
    history = np.array(st.session_state["pred_history"])
    return history.mean(axis=0)

def annotate_frame(frame_bgr, label, confidence, alert, current_fps):
    """Draw detection overlay on the frame."""
    h, w = frame_bgr.shape[:2]

    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    color = (60, 80, 255) if label == "Drowsy" else (60, 220, 100)

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame_bgr, (fx, fy), (fx+fw, fy+fh), color, 2)
        label_text = f"{label}  {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame_bgr, (fx, fy-th-14), (fx+tw+10, fy), color, -1)
        cv2.putText(frame_bgr, label_text, (fx+5, fy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        eye_roi = gray[fy:fy+int(fh*0.6), fx:fx+fw]
        eyes = EYE_CASCADE.detectMultiScale(eye_roi, 1.1, 3)
        for (ex, ey, ew, eh) in eyes[:2]:
            cv2.rectangle(frame_bgr, (fx+ex, fy+ey), (fx+ex+ew, fy+ey+eh), (255, 200, 60), 1)
        break

    if alert:
        cv2.rectangle(frame_bgr, (0, 0), (w-1, h-1), (0, 0, 255), 8)
        cv2.putText(frame_bgr, "! DROWSINESS ALERT !", (w//2 - 160, h-16), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)

    hud_col = (60, 80, 255) if alert else (60, 220, 100)
    cv2.putText(frame_bgr, "DROWSYGUARD", (12, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, hud_col, 2)
    cv2.putText(frame_bgr, f"FPS {current_fps}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1)

    return frame_bgr

# ─────────────────────────────────────────────────────────────
# WebRTC Audio/Video Frame Processor Callback
# ─────────────────────────────────────────────────────────────
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert incoming browser video package to raw BGR for OpenCV matching your logic
    img_bgr = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    t0 = time.time()
    if st.session_state["start_time"] is None:
        st.session_state["start_time"] = t0

    # ── Inference Execution ──
    raw_probs = predict_frame(img_rgb)
    smooth_probs = smooth_prediction(raw_probs)

    drowsy_conf = float(smooth_probs[DROWSY_IDX])

    # Dynamic Threshold filtering
    if max(smooth_probs) >= st.session_state["conf_thresh_val"]:
        pred_idx = int(np.argmax(smooth_probs))
        label = CLASSES[pred_idx]
        conf = float(smooth_probs[pred_idx])
    else:
        label = st.session_state["state_label"] if st.session_state["state_label"] != "—" else "Non Drowsy"
        conf = max(smooth_probs)

    # ── Sequential Streak Analysis Logic ──
    if label == "Drowsy":
        st.session_state["drowsy_streak"] += 1
    else:
        st.session_state["drowsy_streak"] = max(0, st.session_state["drowsy_streak"] - 1)

    was_alert = st.session_state["alert"]
    alert = st.session_state["drowsy_streak"] >= st.session_state["alert_frames_val"]

    if alert and not was_alert:
        st.session_state["alert_count"] += 1
        st.session_state["alert_msg"] = f"🚨 DROWSINESS DETECTED — confidence {drowsy_conf:.0%}"
        log_event(f"ALERT: drowsy for {st.session_state['alert_frames_val']} frames ({drowsy_conf:.0%} conf)", "alert")
    elif not alert and was_alert:
        st.session_state["alert_msg"] = ""
        log_event("Alert cleared — driver alert", "ok")

    # Update global metric references safely from back-thread wrapper
    current_fps = max(1, int(1.0 / max(time.time() - t0, 0.001)))
    st.session_state["alert"] = alert
    st.session_state["state_label"] = label
    st.session_state["confidence"] = conf
    st.session_state["fps"] = current_fps
    st.session_state["frame_count"] += 1
    st.session_state["conf_history"].append((conf, label))

    # Compile annotations directly back to output pipeline arrays
    annotated_bgr = annotate_frame(img_bgr, label, conf, alert, current_fps)
    
    return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

# ─────────────────────────────────────────────────────────────
# Sidebar Configurations
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_path_input = st.text_input("Model .h5 path", value=MODEL_PATH)
    
    # Sliders directly write to explicit state bindings referenced in callbacks
    alert_frames = st.slider("Alert sensitivity (frames)", 5, 30, ALERT_FRAMES,
                             help="Lower = more sensitive. At ~15 fps, 12 frames ≈ 0.8s")
    conf_thresh = st.slider("Min confidence threshold", 0.5, 0.95, CONF_THRESH, 0.05,
                             help="Predictions below this are ignored")
    smooth_win = st.slider("Temporal smoothing window", 1, 15, SMOOTH_WINDOW,
                            help="Average N frames — reduces jitter")

    st.session_state["alert_frames_val"] = alert_frames
    st.session_state["conf_thresh_val"] = conf_thresh
    
    if len(st.session_state["pred_history"]) != smooth_win:
        st.session_state["pred_history"] = collections.deque(maxlen=smooth_win)

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
# Page header HTML Layout
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
# Structural Multi-Column Layout
# ─────────────────────────────────────────────────────────────
col_cam, col_panel = st.columns([5, 3], gap="large")

with col_cam:
    banner_ph = st.empty()
    
    # ── Integrated WebRTC Streamer Window Element ──
    ctx = webrtc_streamer(
        key="drowsy-guard-streamer",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    
    # State reset context management rules checking live updates
    if ctx.state.playing:
        if st.session_state["start_time"] is None:
            st.session_state["start_time"] = time.time()
    else:
        st.session_state["start_time"] = None
        st.session_state["alert"] = False
        st.session_state["state_label"] = "—"
        st.session_state["confidence"] = 0.0

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
# Dynamic Component Panel Update Processing Loops
# ─────────────────────────────────────────────────────────────
def render_panel_ui():
    label = st.session_state["state_label"]
    conf  = st.session_state["confidence"]
    alert = st.session_state["alert"]

    pill_cls = "pill-alert" if label == "Drowsy" else ("pill-ok" if label == "Non Drowsy" else "pill-wait")
    conf_pct = int(conf * 100)
    bar_color = "#ef4444" if label == "Drowsy" else ("#22c55e" if label == "Non Drowsy" else "#334155")

    state_ph.markdown(f"""
    <div style="margin-bottom:6px">
      <span class="pill {pill_cls}">{label if label != '—' else '— waiting —'}</span>
    </div>""", unsafe_allow_html=True)

    conf_ph.markdown(f"""
    <div class="big-conf">
      <div class="num" style="color:{bar_color}">{conf_pct}%</div>
      <div class="lbl">model confidence</div>
      <div class="pbar-track">
        <div class="pbar-fill" style="width:{conf_pct}%;background:{bar_color}"></div>
      </div>
    </div>""", unsafe_allow_html=True)

    fps_ph.metric("FPS",    st.session_state["fps"])
    alerts_ph.metric("Alerts", st.session_state["alert_count"])
    frames_ph.metric("Frames", st.session_state["frame_count"])

    if st.session_state["start_time"]:
        s = int(time.time() - st.session_state["start_time"])
        uptime_ph.metric("Uptime", f"{s//60:02d}:{s%60:02d}")
    else:
        uptime_ph.metric("Uptime", "—")

    # Banner generation checks
    if ctx.state.playing:
        if alert:
            banner_ph.markdown(f'<div class="alert-banner">{st.session_state["alert_msg"]}</div>', unsafe_allow_html=True)
        else:
            banner_ph.markdown(f'<div class="ok-banner">✅ &nbsp; Driver Alert &nbsp;·&nbsp; {label} &nbsp;({conf:.0%})</div>', unsafe_allow_html=True)
    else:
        banner_ph.markdown('<div class="ok-banner" style="background:#1e2535;color:#64748b;border-color:#334155">📷 Camera Stopped. Click "Start" above to stream window.</div>', unsafe_allow_html=True)

    # Sparkline chart array updates
    history = list(st.session_state["conf_history"])
    if history:
        bars_html = ""
        for i, (c, lbl) in enumerate(history):
            ht  = max(4, int(c * 48))
            clr = "#ef4444" if lbl == "Drowsy" else "#22c55e"
            bars_html += f'<div class="hist-bar" style="height:{ht}px;background:{clr}"></div>'
        hist_ph.markdown(f'<div class="hist-wrap">{bars_html}</div>', unsafe_allow_html=True)

    # Event log rendering
    log_html = ""
    for ts, msg, kind in st.session_state["event_log"][:10]:
        cls = {"ok": "lo", "warn": "lw", "alert": "la"}.get(kind, "lo")
        log_html += f'<div class="logline"><span class="lt">{ts}</span><span class="{cls}">{msg}</span></div>'
    log_ph.markdown(log_html or '<div class="logline"><span class="lt">—</span><span class="lo">Waiting…</span></div>', unsafe_allow_html=True)

# Continuous auto-refresh trigger to paint the frame analytics while WebRTC thread processes frames
if ctx.state.playing:
    time.sleep(0.1)
    st.rerun()
else:
    render_panel_ui()
