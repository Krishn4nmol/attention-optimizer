import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import vlc
import os
from collections import deque
from stable_baselines3 import PPO
import plotly.graph_objects as go

st.set_page_config(
    page_title="Attention VLC Player",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2e3250;
    }
    .attention-high { color: #00ff88; font-size: 42px; font-weight: bold; }
    .attention-mid  { color: #ffaa00; font-size: 42px; font-weight: bold; }
    .attention-low  { color: #ff4444; font-size: 42px; font-weight: bold; }
    .status-box {
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 10px;
    }
    .status-playing  { background:#1a3a2a; border:2px solid #00ff88; color:#00ff88; }
    .status-paused   { background:#3a1a1a; border:2px solid #ff4444; color:#ff4444; }
    .status-slowdown { background:#3a2a1a; border:2px solid #ffaa00; color:#ffaa00; }
    .status-break    { background:#1a1a3a; border:2px solid #4488ff; color:#4488ff; }
</style>
""", unsafe_allow_html=True)

# ── MediaPipe ──────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# ── Session State ──────────────────────────────────────────────
defaults = {
    "running":           False,
    "attention_history": deque(maxlen=150),
    "time_history":      deque(maxlen=150),
    "blink_count":       0,
    "session_start":     None,
    "step":              0,
    "pause_count":       0,
    "break_count":       0,
    "video_path":        None,
    "current_action":    "Not Started",
    "action_counts": {
        "Continue": 0, "Pause Video": 0,
        "Slow Down": 0, "Simplify Content": 0, "Add Break": 0
    }
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ────────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
           for i in eye_indices]
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    hd = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (v1 + v2) / (2.0 * hd + 1e-6)

def gaze_score(landmarks, eye_idx, iris_idx, w, h):
    eye_pts  = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx]
    iris_pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in iris_idx]
    eye_cx   = (eye_pts[0][0] + eye_pts[3][0]) / 2
    iris_cx  = np.mean([p[0] for p in iris_pts])
    eye_w    = abs(eye_pts[0][0] - eye_pts[3][0]) + 1e-6
    offset   = abs(iris_cx - eye_cx) / eye_w
    return max(0.0, 1.0 - offset * 3)

def compute_attention(landmarks, w, h):
    l_ear    = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
    r_ear    = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    avg_ear  = (l_ear + r_ear) / 2.0
    l_gaze   = gaze_score(landmarks, LEFT_EYE,  LEFT_IRIS,  w, h)
    r_gaze   = gaze_score(landmarks, RIGHT_EYE, RIGHT_IRIS, w, h)
    avg_gaze = (l_gaze + r_gaze) / 2.0
    nose     = landmarks[4]
    head_s   = max(0.0, 1.0 - abs(nose.x - 0.5) * 2.5)
    ear_s    = min(1.0, avg_ear / 0.3)
    score    = 0.40 * ear_s + 0.40 * avg_gaze + 0.20 * head_s
    return round(float(np.clip(score, 0.0, 1.0)), 2), avg_ear

@st.cache_resource
def load_model():
    try:
        return PPO.load("models/attention_ppo_final")
    except:
        return None

def get_status_html(action_name, score):
    pct = int(score * 100)
    mapping = {
        "Continue":         ("status-playing",  f"▶️ Playing — {pct}%"),
        "Slow Down":        ("status-slowdown", f"🐢 Slow Mode — {pct}%"),
        "Pause Video":      ("status-paused",   f"⏸️ Auto Paused — {pct}%"),
        "Simplify Content": ("status-slowdown", f"📖 Rewinding — {pct}%"),
        "Add Break":        ("status-break",    f"☕ Break Time — {pct}%"),
        "Not Started":      ("status-playing",  "⬆️ Press Start"),
    }
    css, label = mapping.get(action_name, ("status-playing", action_name))
    return f'<div class="status-box {css}">{label}</div>'

# ── UI ─────────────────────────────────────────────────────────
st.title("🎬 Attention-Adaptive VLC Player")
st.markdown("*VLC plays your video — AI controls pause, speed and rewind automatically*")
st.divider()

st.subheader("📁 Step 1: Select Your Lecture Video")
video_path_input = st.text_input(
    "Video path",
    value=r"C:\Users\KIIT\OneDrive\Documents\attention-optimizer\lecture.mp4",
    placeholder=r"C:\path\to\your\video.mp4"
)

if video_path_input and os.path.exists(video_path_input):
    st.session_state.video_path = video_path_input
    st.success(f"✅ Found: {os.path.basename(video_path_input)}")
elif video_path_input:
    st.error("❌ File not found — check the path")

st.divider()

col_start, col_stop, col_status = st.columns([1, 1, 2])
with col_start:
    start_btn = st.button(
        "▶ Start Session",
        type="primary",
        disabled=not bool(st.session_state.video_path)
    )
with col_stop:
    stop_btn = st.button("⏹ Stop Session")
with col_status:
    status_placeholder = st.empty()

st.divider()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📷 Webcam — AI Watching")
    webcam_placeholder    = st.empty()
    st.subheader("🎬 VLC Status")
    vlc_status_placeholder = st.empty()
    st.subheader("👁️ Attention Score")
    score_placeholder     = st.empty()

with col_right:
    st.subheader("📊 Session Stats")
    stats_placeholder        = st.empty()
    st.subheader("📈 Attention Graph")
    chart_placeholder        = st.empty()
    st.subheader("🎯 Action Distribution")
    action_chart_placeholder = st.empty()

# ── Start/Stop ─────────────────────────────────────────────────
if start_btn:
    st.session_state.running        = True
    st.session_state.session_start  = time.time()
    st.session_state.attention_history.clear()
    st.session_state.time_history.clear()
    st.session_state.blink_count    = 0
    st.session_state.step           = 0
    st.session_state.pause_count    = 0
    st.session_state.break_count    = 0
    st.session_state.current_action = "Not Started"
    st.session_state.action_counts  = {
        "Continue": 0, "Pause Video": 0,
        "Slow Down": 0, "Simplify Content": 0, "Add Break": 0
    }

if stop_btn:
    st.session_state.running = False

# ── Main Loop ──────────────────────────────────────────────────
if st.session_state.running and st.session_state.video_path:
    model   = load_model()
    cam_cap = cv2.VideoCapture(0)

    # Start VLC
    vlc_instance = vlc.Instance("--no-xlib")
    media        = vlc_instance.media_new(st.session_state.video_path)
    player       = vlc_instance.media_player_new()
    player.set_media(media)
    player.play()
    time.sleep(1.5)

    EAR_THRESHOLD  = 0.22
    blink_cooldown = 0
    prev_attention = 1.0
    low_streak     = 0
    no_face_frames = 0
    last_action    = "Continue"

    action_names = [
        "Continue", "Pause Video", "Slow Down",
        "Simplify Content", "Add Break"
    ]

    status_placeholder.success("🟢 VLC playing — AI monitoring attention!")

    while st.session_state.running:
        step = st.session_state.step

        ret, cam_frame = cam_cap.read()
        if not ret:
            break

        h, w    = cam_frame.shape[:2]
        rgb     = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            no_face_frames = 0
            lms = results.multi_face_landmarks[0].landmark
            attention_score, avg_ear = compute_attention(lms, w, h)
            if avg_ear < EAR_THRESHOLD and blink_cooldown == 0:
                st.session_state.blink_count += 1
                blink_cooldown = 10
            if blink_cooldown > 0:
                blink_cooldown -= 1
            color = (0,255,0) if attention_score>0.65 else \
                    (0,165,255) if attention_score>0.40 else (0,0,255)
            cv2.putText(cam_frame, f"{int(attention_score*100)}%",
                        (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(cam_frame, last_action,
                        (20,95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            no_face_frames += 1
            drop = min(0.05 * no_face_frames, 0.15)
            attention_score = max(0.0, prev_attention - drop)
            cv2.putText(cam_frame, "No face",
                        (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.putText(cam_frame, f"{int(attention_score*100)}%",
                        (20,90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)

        if attention_score < 0.45:
            low_streak += 1
        else:
            low_streak = max(0, low_streak - 1)

        trend = float(np.clip((attention_score - prev_attention)*5, -1, 1))
        obs = np.array([
            attention_score, trend,
            min(step/50.0, 1.0),
            min(low_streak/10.0, 1.0),
            min(step/300.0, 1.0), 0.5
        ], dtype=np.float32)

        if model:
            action, _   = model.predict(obs, deterministic=True)
            action_name = action_names[int(action)]
        else:
            if attention_score >= 0.65:   action_name = "Continue"
            elif attention_score >= 0.45: action_name = "Slow Down"
            elif attention_score >= 0.30: action_name = "Pause Video"
            else:                         action_name = "Add Break"

        # ── VLC Control (only on action change) ───────────────
        if action_name != last_action:
            if action_name == "Continue":
                player.set_rate(1.0)
                player.play()

            elif action_name == "Slow Down":
                player.set_rate(0.6)
                player.play()

            elif action_name == "Pause Video":
                player.pause()

            elif action_name == "Simplify Content":
                current_ms = player.get_time()
                player.set_time(max(0, current_ms - 15000))
                player.set_rate(0.5)
                player.play()

            elif action_name == "Add Break":
                player.pause()

            last_action = action_name

        # Resume if refocused
        if action_name == "Continue" and \
           player.get_state() == vlc.State.Paused:
            player.play()
            player.set_rate(1.0)

        # ── Update counts ──────────────────────────────────────
        st.session_state.current_action = action_name
        st.session_state.action_counts[action_name] += 1

        # Sync pause and break counts from action_counts
        st.session_state.pause_count = \
            st.session_state.action_counts["Pause Video"] // 10
        st.session_state.break_count = \
            st.session_state.action_counts["Add Break"] // 10

        elapsed = time.time() - st.session_state.session_start
        st.session_state.attention_history.append(attention_score)
        st.session_state.time_history.append(round(elapsed, 1))
        prev_attention = attention_score
        st.session_state.step += 1

        # ── Update UI ─────────────────────────────────────────
        webcam_placeholder.image(
            cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB),
            channels="RGB", width=400
        )

        vlc_status_placeholder.markdown(
            get_status_html(action_name, attention_score),
            unsafe_allow_html=True
        )

        score_pct = int(attention_score * 100)
        css_class = "attention-high" if attention_score > 0.65 else \
                    "attention-mid"  if attention_score > 0.40 else \
                    "attention-low"
        score_placeholder.markdown(
            f'<div class="metric-card">'
            f'<div style="color:#aaa;font-size:12px">LIVE SCORE</div>'
            f'<div class="{css_class}">{score_pct}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        vlc_state = str(player.get_state()).replace("State.", "")
        vid_time  = player.get_time() // 1000
        vm = vid_time // 60
        vs = vid_time % 60

        stats_placeholder.markdown(
            f'<div class="metric-card">'
            f'<div style="color:#aaa;font-size:12px">SESSION TIME</div>'
            f'<div style="color:white;font-size:22px">{mins:02d}:{secs:02d}</div>'
            f'<br>'
            f'<div style="color:#aaa;font-size:12px">VIDEO TIME</div>'
            f'<div style="color:white;font-size:22px">{vm:02d}:{vs:02d}</div>'
            f'<br>'
            f'<div style="color:#aaa;font-size:12px">VLC STATE</div>'
            f'<div style="color:#00ff88;font-size:18px">{vlc_state}</div>'
            f'<br>'
            f'<div style="color:#aaa;font-size:12px">AUTO PAUSES</div>'
            f'<div style="color:#ff4444;font-size:22px">'
            f'{st.session_state.pause_count}</div>'
            f'<br>'
            f'<div style="color:#aaa;font-size:12px">BREAKS</div>'
            f'<div style="color:#4488ff;font-size:22px">'
            f'{st.session_state.break_count}</div>'
            f'<br>'
            f'<div style="color:#aaa;font-size:12px">BLINKS</div>'
            f'<div style="color:white;font-size:22px">'
            f'{st.session_state.blink_count}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        if len(st.session_state.attention_history) > 1:
            times  = list(st.session_state.time_history)
            scores = list(st.session_state.attention_history)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times, y=scores, mode="lines",
                line=dict(color="#00ff88", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,255,136,0.1)"
            ))
            fig.add_hline(y=0.65, line_dash="dash", line_color="#ffaa00")
            fig.add_hline(y=0.40, line_dash="dash", line_color="#ff4444")
            fig.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=5, b=0),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="white"),
                yaxis=dict(range=[0,1], gridcolor="#2e3250"),
                xaxis=dict(gridcolor="#2e3250"),
                showlegend=False
            )
            chart_placeholder.plotly_chart(
                fig, use_container_width=True,
                key=f"chart_{step}"
            )

        ac   = st.session_state.action_counts
        fig2 = go.Figure(go.Bar(
            x=list(ac.keys()),
            y=list(ac.values()),
            marker_color=["#00ff88","#ff4444","#ffaa00","#4488ff","#ff88aa"]
        ))
        fig2.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=5, b=0),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
            yaxis=dict(gridcolor="#2e3250"),
            xaxis=dict(gridcolor="#2e3250"),
        )
        action_chart_placeholder.plotly_chart(
            fig2, use_container_width=True,
            key=f"action_chart_{step}"
        )

        time.sleep(0.1)

    player.stop()
    cam_cap.release()
    status_placeholder.warning("⏹ Session ended")

elif not st.session_state.running:
    status_placeholder.info("⬆️ Enter video path and press Start Session")