import cv2
import mediapipe as mp
import numpy as np
import time

# ─── MediaPipe Setup ───────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ─── Eye Landmark Indices ──────────────────────────────────────
# Left eye corners and top/bottom points
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Iris center landmarks (needs refine_landmarks=True)
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# ─── Helper Functions ──────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h):
    """
    EAR = how open your eye is.
    High EAR = eye open = awake
    Low EAR  = eye closed = blinking or sleepy
    """
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        points.append((x, y))

    # Vertical distances
    v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    # Horizontal distance
    h  = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    ear = (v1 + v2) / (2.0 * h + 1e-6)
    return ear

def gaze_score(landmarks, eye_indices, iris_indices, frame_w, frame_h):
    """
    How centered is your iris inside your eye.
    Centered = looking at screen = focused
    Far left/right = looking away = distracted
    """
    eye_pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        eye_pts.append((lm.x * frame_w, lm.y * frame_h))

    iris_pts = []
    for idx in iris_indices:
        lm = landmarks[idx]
        iris_pts.append((lm.x * frame_w, lm.y * frame_h))

    eye_center_x  = (eye_pts[0][0] + eye_pts[3][0]) / 2
    iris_center_x = np.mean([p[0] for p in iris_pts])

    eye_width = abs(eye_pts[0][0] - eye_pts[3][0]) + 1e-6
    offset    = abs(iris_center_x - eye_center_x) / eye_width

    # offset near 0 = centered = good
    # offset near 0.5 = looking away = bad
    score = max(0.0, 1.0 - (offset * 3))
    return score

# ─── Main Loop ─────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("Starting Attention Monitor... Press Q to quit")
print("-" * 40)

# Blink tracking
blink_counter    = 0
blink_cooldown   = 0
EAR_THRESHOLD    = 0.22   # below this = eye closed
attention_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    attention_score = 0.5  # default if no face detected

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # 1. Eye Aspect Ratio (both eyes)
        left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE,  frame_w, frame_h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, frame_w, frame_h)
        avg_ear   = (left_ear + right_ear) / 2.0

        # 2. Blink detection
        if avg_ear < EAR_THRESHOLD:
            if blink_cooldown == 0:
                blink_counter += 1
                blink_cooldown = 10  # frames cooldown
        if blink_cooldown > 0:
            blink_cooldown -= 1

        # 3. Gaze Score (both eyes)
        left_gaze  = gaze_score(landmarks, LEFT_EYE,  LEFT_IRIS,  frame_w, frame_h)
        right_gaze = gaze_score(landmarks, RIGHT_EYE, RIGHT_IRIS, frame_w, frame_h)
        avg_gaze   = (left_gaze + right_gaze) / 2.0

        # 4. Head pose (are they looking away entirely?)
        nose_tip  = landmarks[4]
        head_turn = abs(nose_tip.x - 0.5)  # 0=center, 0.5=fully turned
        head_score = max(0.0, 1.0 - (head_turn * 2.5))

        # 5. Combine into final attention score
        # EAR contributes 40%, gaze 40%, head pose 20%
        ear_score = min(1.0, avg_ear / 0.3)  # normalize EAR to 0-1
        attention_score = (
            0.40 * ear_score +
            0.40 * avg_gaze  +
            0.20 * head_score
        )
        attention_score = round(float(np.clip(attention_score, 0.0, 1.0)), 2)
        attention_history.append(attention_score)

        # ── Display on frame ──────────────────────────────────
        # Color changes based on attention
        if attention_score > 0.65:
            color = (0, 255, 0)    # Green  = focused
            label = "FOCUSED"
        elif attention_score > 0.40:
            color = (0, 165, 255)  # Orange = moderate
            label = "MODERATE"
        else:
            color = (0, 0, 255)    # Red    = distracted
            label = "DISTRACTED"

        cv2.putText(frame, f"Attention: {attention_score}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, label,
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"Blinks: {blink_counter}",
                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Gaze:  {avg_gaze:.2f}",
                    (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Head:  {head_score:.2f}",
                    (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Print to terminal too
        print(f"Attention: {attention_score:.2f} | "
              f"EAR: {avg_ear:.2f} | "
              f"Gaze: {avg_gaze:.2f} | "
              f"Head: {head_score:.2f} | "
              f"Blinks: {blink_counter}")

    else:
        cv2.putText(frame, "No face detected",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    cv2.imshow("Attention Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ─── Summary ───────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

if attention_history:
    print("\n" + "="*40)
    print("SESSION SUMMARY")
    print("="*40)
    print(f"Average Attention : {np.mean(attention_history):.2f}")
    print(f"Min Attention     : {np.min(attention_history):.2f}")
    print(f"Max Attention     : {np.max(attention_history):.2f}")
    print(f"Total Blinks      : {blink_counter}")
    print("="*40)