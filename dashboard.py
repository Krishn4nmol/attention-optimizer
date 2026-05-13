import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from stable_baselines3 import PPO
from attention_env import AttentionEnv
import plotly.graph_objects as go

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Attention Optimizer",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────
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
    .attention-high  { color: #00ff88; font-size: 48px; font-weight: bold; }
    .attention-mid   { color: #ffaa00; font-size: 48px; font-weight: bold; }
    .attention-low   { color: #ff4444; font-size: 48px; font-weight: bold; }
    .action-box {
        background: #1e2130;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        font-size: 24px;
        border: 2px solid #3e4470;
        margin-top: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-size: 18px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── MediaPipe Setup ────────────────────────────────────────────
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
if "running" not in st.session_state:
    st.session_state.running = False
if "attention_history" not in st.session_state:
    st.session_state.attention_history = deque(maxlen=100)
if "time_history" not in st.session_state:
    st.session_state.time_history = deque(maxlen=100)
if "current_score" not in st.session_state:
    st.session_state.current_score = 0.0
if "current_action" not in st.session_state:
    st.session_state.current_action = "Not Started"
if "blink_count" not in st.session_state:
    st.session_state.blink_count = 0
if "session_start" not in st.session_state:
    st.session_state.session_start = None
if "action_counts" not in st.session_state:
    st.session_state.action_counts = {
        "Continue": 0, "Pause Video": 0,
        "Slow Down": 0, "Simplify Content": 0, "Add Break": 0
    }
if "step" not in st.session_state:
    st.session_state.step = 0

# ── Helper Functions ───────────────────────────────────────────
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

# ── Load PPO Model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return PPO.load("models/attention_ppo_final")
    except:
        return None

# ── UI Layout ──────────────────────────────────────────────────
st.title("🧠 Attention Span Optimizer")
st.markdown("*Real-time attention monitoring with AI-powered content adaptation*")
st.divider()

col_start, col_stop, col_status = st.columns([1, 1, 2])
with col_start:
    start_btn = st.button("▶ Start Session", type="primary")
with col_stop:
    stop_btn = st.button("⏹ Stop Session")
with col_status:
    status_placeholder = st.empty()

st.divider()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📷 Live Feed")
    webcam_placeholder = st.empty()
    st.subheader("📊 Current Stats")
    score_placeholder  = st.empty()
    action_placeholder = st.empty()
    blink_placeholder  = st.empty()

with col_right:
    st.subheader("📈 Attention Over Time")
    chart_placeholder = st.empty()
    st.subheader("🎯 Action Distribution")
    action_chart_placeholder = st.empty()

# ── Session Logic ──────────────────────────────────────────────
if start_btn:
    st.session_state.running       = True
    st.session_state.session_start = time.time()
    st.session_state.attention_history.clear()
    st.session_state.time_history.clear()
    st.session_state.blink_count   = 0
    st.session_state.step          = 0
    st.session_state.action_counts = {
        "Continue": 0, "Pause Video": 0,
        "Slow Down": 0, "Simplify Content": 0, "Add Break": 0
    }

if stop_btn:
    st.session_state.running = False

# ── Main Loop ──────────────────────────────────────────────────
if st.session_state.running:
    model  = load_model()
    cap    = cv2.VideoCapture(0)

    EAR_THRESHOLD  = 0.22
    blink_cooldown = 0
    prev_attention = 1.0
    low_streak     = 0

    action_names  = [
        "Continue", "Pause Video", "Slow Down",
        "Simplify Content", "Add Break"
    ]
    action_emojis = {
        "Continue": "▶️", "Pause Video": "⏸️",
        "Slow Down": "🐢", "Simplify Content": "📖",
        "Add Break": "☕"
    }

    status_placeholder.success("🟢 Session Running — Look at your screen!")

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        h, w    = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        attention_score = prev_attention
        step = st.session_state.step

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            attention_score, avg_ear = compute_attention(lms, w, h)

            # Blink detection
            if avg_ear < EAR_THRESHOLD and blink_cooldown == 0:
                st.session_state.blink_count += 1
                blink_cooldown = 10
            if blink_cooldown > 0:
                blink_cooldown -= 1

            # Low streak tracking
            if attention_score < 0.45:
                low_streak += 1
            else:
                low_streak = max(0, low_streak - 1)

            # Build observation for PPO
            trend = float(np.clip(
                (attention_score - prev_attention) * 5, -1, 1
            ))
            obs = np.array([
                attention_score,
                trend,
                min(step / 50.0,  1.0),
                min(low_streak / 10.0, 1.0),
                min(step / 300.0, 1.0),
                0.5
            ], dtype=np.float32)

            # PPO decision
            if model:
                action, _   = model.predict(obs, deterministic=True)
                action_name = action_names[int(action)]
            else:
                if attention_score >= 0.65:
                    action_name = "Continue"
                elif attention_score >= 0.45:
                    action_name = "Slow Down"
                elif attention_score >= 0.30:
                    action_name = "Pause Video"
                else:
                    action_name = "Add Break"

            st.session_state.current_action = action_name
            st.session_state.action_counts[action_name] += 1

            # Draw on frame
            color = (0, 255, 0)   if attention_score > 0.65 else \
                    (0, 165, 255) if attention_score > 0.40 else \
                    (0, 0, 255)
            cv2.putText(frame, f"Attention: {attention_score:.2f}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, action_name,
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Update history
        elapsed = time.time() - st.session_state.session_start
        st.session_state.attention_history.append(attention_score)
        st.session_state.time_history.append(round(elapsed, 1))
        st.session_state.current_score = attention_score
        prev_attention = attention_score
        st.session_state.step += 1

        # ── Update UI ─────────────────────────────────────────

        # Webcam
        webcam_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            width=450
        )

        # Score
        score_pct = int(attention_score * 100)
        css_class = "attention-high" if attention_score > 0.65 else \
                    "attention-mid"  if attention_score > 0.40 else \
                    "attention-low"
        score_placeholder.markdown(
            f'<div class="metric-card">'
            f'<div style="color:#aaa;font-size:14px">ATTENTION SCORE</div>'
            f'<div class="{css_class}">{score_pct}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Action
        emoji = action_emojis.get(st.session_state.current_action, "▶️")
        action_placeholder.markdown(
            f'<div class="action-box">'
            f'{emoji} {st.session_state.current_action}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Blinks
        blink_placeholder.metric("👁️ Blinks", st.session_state.blink_count)

        # Attention chart
        if len(st.session_state.attention_history) > 1:
            times  = list(st.session_state.time_history)
            scores = list(st.session_state.attention_history)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times, y=scores,
                mode="lines",
                line=dict(color="#00ff88", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,255,136,0.1)",
                name="Attention"
            ))
            fig.add_hline(
                y=0.65, line_dash="dash",
                line_color="#ffaa00",
                annotation_text="Focus threshold"
            )
            fig.add_hline(
                y=0.40, line_dash="dash",
                line_color="#ff4444",
                annotation_text="Critical threshold"
            )
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="white"),
                yaxis=dict(range=[0, 1], gridcolor="#2e3250"),
                xaxis=dict(title="Time (s)", gridcolor="#2e3250"),
                showlegend=False
            )
            chart_placeholder.plotly_chart(
                fig,
                use_container_width=True,
                key=f"attention_chart_{step}"
            )

        # Action distribution chart
        ac   = st.session_state.action_counts
        fig2 = go.Figure(go.Bar(
            x=list(ac.keys()),
            y=list(ac.values()),
            marker_color=[
                "#00ff88", "#ff4444", "#ffaa00",
                "#4488ff", "#ff88aa"
            ]
        ))
        fig2.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
            yaxis=dict(gridcolor="#2e3250"),
            xaxis=dict(gridcolor="#2e3250"),
        )
        action_chart_placeholder.plotly_chart(
            fig2,
            use_container_width=True,
            key=f"action_chart_{step}"
        )

        time.sleep(0.1)

    cap.release()
    status_placeholder.warning("⏹ Session Stopped")

elif not st.session_state.running:
    status_placeholder.info("⬆️ Press Start Session to begin")
    webcam_placeholder.markdown(
        '<div style="background:#1e2130;height:300px;border-radius:12px;'
        'display:flex;align-items:center;justify-content:center;'
        'color:#aaa;font-size:18px;">📷 Camera will appear here</div>',
        unsafe_allow_html=True
    )