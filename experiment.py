import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from collections import deque
from stable_baselines3 import PPO
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Experiment Module",
    page_icon="🔬",
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
    .exp-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2e3250;
        margin: 10px 0;
    }
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
    "exp_running":        False,
    "exp_mode":           "adaptive",  # adaptive or baseline
    "exp_phase":          "idle",      # idle, running, done
    "adaptive_data":      [],
    "baseline_data":      [],
    "current_session":    [],
    "session_start":      None,
    "step":               0,
    "participant_id":     "P001",
    "session_duration":   120,  # seconds
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

def save_to_csv(data, filename):
    os.makedirs("results", exist_ok=True)
    filepath = f"results/{filename}"
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    return filepath

def compute_metrics(data):
    if not data:
        return {}
    scores = [d["attention_score"] for d in data]
    actions = [d["action"] for d in data]
    return {
        "avg_attention":    round(np.mean(scores), 3),
        "min_attention":    round(np.min(scores), 3),
        "max_attention":    round(np.max(scores), 3),
        "std_attention":    round(np.std(scores), 3),
        "time_focused":     round(sum(1 for s in scores if s > 0.65) / len(scores) * 100, 1),
        "time_moderate":    round(sum(1 for s in scores if 0.40 < s <= 0.65) / len(scores) * 100, 1),
        "time_distracted":  round(sum(1 for s in scores if s <= 0.40) / len(scores) * 100, 1),
        "total_pauses":     actions.count("Pause Video"),
        "total_breaks":     actions.count("Add Break"),
        "total_slowdowns":  actions.count("Slow Down"),
        "attention_drops":  sum(1 for i in range(1, len(scores))
                               if scores[i] < 0.45 and scores[i-1] >= 0.45),
    }

# ── UI ─────────────────────────────────────────────────────────
st.title("🔬 Experiment Module")
st.markdown("*Compare adaptive vs non-adaptive learning — generates IEEE paper results*")
st.divider()

# Experiment Setup
col1, col2, col3 = st.columns(3)
with col1:
    participant_id = st.text_input(
        "Participant ID",
        value=st.session_state.participant_id
    )
    st.session_state.participant_id = participant_id

with col2:
    duration = st.selectbox(
        "Session Duration",
        [60, 120, 180, 300],
        index=1,
        format_func=lambda x: f"{x//60} min {x%60} sec" if x%60
                               else f"{x//60} minutes"
    )
    st.session_state.session_duration = duration

with col3:
    mode = st.selectbox(
        "Session Mode",
        ["adaptive", "baseline"],
        format_func=lambda x: "🤖 Adaptive (AI)" if x == "adaptive"
                              else "📺 Baseline (No AI)"
    )
    st.session_state.exp_mode = mode

st.divider()

# Mode explanation
if st.session_state.exp_mode == "adaptive":
    st.info("🤖 **Adaptive Mode** — AI monitors attention and adapts content delivery in real time")
else:
    st.warning("📺 **Baseline Mode** — No adaptation. Student watches without any AI intervention. Used as control condition.")

st.divider()

# Controls
col_start, col_stop = st.columns([1, 1])
with col_start:
    start_btn = st.button("▶ Start Experiment", type="primary")
with col_stop:
    stop_btn  = st.button("⏹ Stop Experiment")

status_placeholder = st.empty()
progress_placeholder = st.empty()
st.divider()

# Live display
col_left, col_right = st.columns([1, 2])
with col_left:
    st.subheader("📷 Webcam")
    webcam_placeholder = st.empty()
    st.subheader("👁️ Live Score")
    score_placeholder  = st.empty()
    st.subheader("🎯 Current Action")
    action_placeholder = st.empty()

with col_right:
    st.subheader("📈 Live Attention Graph")
    chart_placeholder  = st.empty()
    st.subheader("📊 Session Stats")
    stats_placeholder  = st.empty()

st.divider()
st.subheader("📋 Results Comparison")
results_placeholder = st.empty()

# ── Start/Stop ─────────────────────────────────────────────────
if start_btn:
    st.session_state.exp_running     = True
    st.session_state.session_start   = time.time()
    st.session_state.current_session = []
    st.session_state.step            = 0

if stop_btn:
    st.session_state.exp_running = False

# ── Experiment Loop ────────────────────────────────────────────
if st.session_state.exp_running:
    model   = load_model()
    cam_cap = cv2.VideoCapture(0)

    attention_history = deque(maxlen=200)
    time_history      = deque(maxlen=200)

    EAR_THRESHOLD  = 0.22
    blink_cooldown = 0
    prev_attention = 1.0
    low_streak     = 0
    no_face_frames = 0

    action_names = [
        "Continue", "Pause Video", "Slow Down",
        "Simplify Content", "Add Break"
    ]

    action_emojis = {
        "Continue":         "▶️",
        "Pause Video":      "⏸️",
        "Slow Down":        "🐢",
        "Simplify Content": "📖",
        "Add Break":        "☕",
        "Baseline":         "📺"
    }

    mode = st.session_state.exp_mode
    status_placeholder.success(
        f"🟢 Running — {'Adaptive' if mode=='adaptive' else 'Baseline'} Mode | "
        f"Participant: {st.session_state.participant_id}"
    )

    while st.session_state.exp_running:
        step    = st.session_state.step
        elapsed = time.time() - st.session_state.session_start

        # Check time limit
        if elapsed >= st.session_state.session_duration:
            st.session_state.exp_running = False
            break

        # Progress bar
        progress = elapsed / st.session_state.session_duration
        remaining = int(st.session_state.session_duration - elapsed)
        rm = remaining // 60
        rs = remaining % 60
        progress_placeholder.progress(
            progress,
            text=f"⏱️ Time remaining: {rm:02d}:{rs:02d}"
        )

        # Webcam
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
                blink_cooldown = 10
            if blink_cooldown > 0:
                blink_cooldown -= 1
            color = (0,255,0) if attention_score>0.65 else \
                    (0,165,255) if attention_score>0.40 else (0,0,255)
            cv2.putText(cam_frame, f"{int(attention_score*100)}%",
                        (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        else:
            no_face_frames += 1
            drop = min(0.05 * no_face_frames, 0.15)
            attention_score = max(0.0, prev_attention - drop)
            cv2.putText(cam_frame, "No face",
                        (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        if attention_score < 0.45:
            low_streak += 1
        else:
            low_streak = max(0, low_streak - 1)

        # Action decision
        if mode == "adaptive" and model:
            trend = float(np.clip(
                (attention_score - prev_attention)*5, -1, 1
            ))
            obs = np.array([
                attention_score, trend,
                min(step/50.0, 1.0),
                min(low_streak/10.0, 1.0),
                min(step/300.0, 1.0), 0.5
            ], dtype=np.float32)
            action, _   = model.predict(obs, deterministic=True)
            action_name = action_names[int(action)]
        else:
            # Baseline — no adaptation
            action_name = "Baseline"

        # Save data point
        st.session_state.current_session.append({
            "participant_id":  st.session_state.participant_id,
            "mode":            mode,
            "timestamp":       round(elapsed, 2),
            "attention_score": attention_score,
            "action":          action_name,
            "low_streak":      low_streak,
        })

        attention_history.append(attention_score)
        time_history.append(round(elapsed, 1))
        prev_attention = attention_score
        st.session_state.step += 1

        # ── Update UI ─────────────────────────────────────────
        webcam_placeholder.image(
            cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB),
            channels="RGB", width=350
        )

        score_pct = int(attention_score * 100)
        css_class = "attention-high" if attention_score > 0.65 else \
                    "attention-mid"  if attention_score > 0.40 else \
                    "attention-low"
        score_placeholder.markdown(
            f'<div class="metric-card">'
            f'<div style="color:#aaa;font-size:12px">ATTENTION</div>'
            f'<div class="{css_class}">{score_pct}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        emoji = action_emojis.get(action_name, "▶️")
        action_placeholder.markdown(
            f'<div class="metric-card">'
            f'<div style="color:#aaa;font-size:12px">ACTION</div>'
            f'<div style="color:white;font-size:22px">{emoji} {action_name}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Live chart
        if len(attention_history) > 1:
            color_mode = "#00ff88" if mode == "adaptive" else "#ffaa00"
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(time_history),
                y=list(attention_history),
                mode="lines",
                line=dict(color=color_mode, width=2),
                fill="tozeroy",
                fillcolor=f"rgba(0,255,136,0.1)" if mode=="adaptive"
                          else "rgba(255,170,0,0.1)",
                name=mode.capitalize()
            ))
            fig.add_hline(y=0.65, line_dash="dash", line_color="#ffaa00")
            fig.add_hline(y=0.40, line_dash="dash", line_color="#ff4444")
            fig.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="white"),
                yaxis=dict(range=[0,1], gridcolor="#2e3250",
                           title="Attention Score"),
                xaxis=dict(gridcolor="#2e3250", title="Time (s)"),
                showlegend=False
            )
            chart_placeholder.plotly_chart(
                fig, use_container_width=True,
                key=f"chart_{step}"
            )

        # Live stats
        if attention_history:
            scores = list(attention_history)
            focused_pct = sum(1 for s in scores if s > 0.65)/len(scores)*100
            stats_placeholder.markdown(
                f'<div class="exp-card">'
                f'<b>Mode:</b> {"🤖 Adaptive" if mode=="adaptive" else "📺 Baseline"}<br><br>'
                f'<b>Avg Attention:</b> {np.mean(scores)*100:.1f}%<br>'
                f'<b>Time Focused:</b> {focused_pct:.1f}%<br>'
                f'<b>Data Points:</b> {len(scores)}<br>'
                f'<b>Elapsed:</b> {int(elapsed)}s / {st.session_state.session_duration}s'
                f'</div>',
                unsafe_allow_html=True
            )

        time.sleep(0.1)

    cam_cap.release()

    # ── Session Complete ───────────────────────────────────────
    if st.session_state.current_session:
        data    = st.session_state.current_session
        metrics = compute_metrics(data)

        # Save to correct list
        if mode == "adaptive":
            st.session_state.adaptive_data = data
        else:
            st.session_state.baseline_data = data

        # Save CSV
        filename = f"{st.session_state.participant_id}_{mode}_{int(time.time())}.csv"
        filepath = save_to_csv(data, filename)

        status_placeholder.success(
            f"✅ Session complete! Saved to {filepath}"
        )
        progress_placeholder.progress(1.0, text="Session Complete!")

# ── Results Comparison ─────────────────────────────────────────
if st.session_state.adaptive_data and st.session_state.baseline_data:
    adaptive_metrics = compute_metrics(st.session_state.adaptive_data)
    baseline_metrics = compute_metrics(st.session_state.baseline_data)

    results_placeholder.markdown("### 📊 Adaptive vs Baseline Comparison")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        diff = adaptive_metrics["avg_attention"] - \
               baseline_metrics["avg_attention"]
        st.metric(
            "Avg Attention",
            f"{adaptive_metrics['avg_attention']*100:.1f}%",
            f"{diff*100:+.1f}% vs baseline"
        )

    with col2:
        diff2 = adaptive_metrics["time_focused"] - \
                baseline_metrics["time_focused"]
        st.metric(
            "Time Focused",
            f"{adaptive_metrics['time_focused']}%",
            f"{diff2:+.1f}% vs baseline"
        )

    with col3:
        diff3 = adaptive_metrics["attention_drops"] - \
                baseline_metrics["attention_drops"]
        st.metric(
            "Attention Drops",
            adaptive_metrics["attention_drops"],
            f"{diff3:+d} vs baseline",
            delta_color="inverse"
        )

    with col4:
        diff4 = adaptive_metrics["std_attention"] - \
                baseline_metrics["std_attention"]
        st.metric(
            "Attention Stability",
            f"{adaptive_metrics['std_attention']:.3f}",
            f"{diff4:+.3f} vs baseline",
            delta_color="inverse"
        )

    # Comparison chart
    st.subheader("📈 Attention Comparison Chart")
    fig = go.Figure()

    adaptive_scores = [d["attention_score"]
                       for d in st.session_state.adaptive_data]
    baseline_scores = [d["attention_score"]
                       for d in st.session_state.baseline_data]
    adaptive_times  = [d["timestamp"]
                       for d in st.session_state.adaptive_data]
    baseline_times  = [d["timestamp"]
                       for d in st.session_state.baseline_data]

    fig.add_trace(go.Scatter(
        x=adaptive_times, y=adaptive_scores,
        mode="lines",
        line=dict(color="#00ff88", width=2),
        name="Adaptive (AI)",
        fill="tozeroy",
        fillcolor="rgba(0,255,136,0.05)"
    ))
    fig.add_trace(go.Scatter(
        x=baseline_times, y=baseline_scores,
        mode="lines",
        line=dict(color="#ffaa00", width=2),
        name="Baseline (No AI)",
        fill="tozeroy",
        fillcolor="rgba(255,170,0,0.05)"
    ))
    fig.add_hline(y=0.65, line_dash="dash", line_color="#ffffff",
                  annotation_text="Focus threshold")
    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        yaxis=dict(range=[0,1], gridcolor="#2e3250",
                   title="Attention Score"),
        xaxis=dict(gridcolor="#2e3250", title="Time (s)"),
        legend=dict(bgcolor="#1e2130")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bar chart comparison
    st.subheader("📊 Metrics Comparison")
    categories = ["Avg Attention", "Time Focused %",
                  "Time Moderate %", "Time Distracted %"]
    adaptive_vals = [
        adaptive_metrics["avg_attention"]*100,
        adaptive_metrics["time_focused"],
        adaptive_metrics["time_moderate"],
        adaptive_metrics["time_distracted"]
    ]
    baseline_vals = [
        baseline_metrics["avg_attention"]*100,
        baseline_metrics["time_focused"],
        baseline_metrics["time_moderate"],
        baseline_metrics["time_distracted"]
    ]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="Adaptive (AI)",
        x=categories,
        y=adaptive_vals,
        marker_color="#00ff88"
    ))
    fig2.add_trace(go.Bar(
        name="Baseline (No AI)",
        x=categories,
        y=baseline_vals,
        marker_color="#ffaa00"
    ))
    fig2.update_layout(
        barmode="group",
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        yaxis=dict(gridcolor="#2e3250"),
        legend=dict(bgcolor="#1e2130")
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Save combined results
    if st.button("💾 Save Full Results to CSV"):
        combined = []
        for d in st.session_state.adaptive_data:
            combined.append(d)
        for d in st.session_state.baseline_data:
            combined.append(d)
        path = save_to_csv(
            combined,
            f"{st.session_state.participant_id}_combined_results.csv"
        )
        st.success(f"✅ Saved to {path}")

elif st.session_state.adaptive_data and not st.session_state.baseline_data:
    st.info("✅ Adaptive session done! Now run a **Baseline** session to compare.")
elif st.session_state.baseline_data and not st.session_state.adaptive_data:
    st.info("✅ Baseline session done! Now run an **Adaptive** session to compare.")
else:
    st.info("Run both an Adaptive and Baseline session to see comparison results.")