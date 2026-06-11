import cv2
import mediapipe as mp
import numpy as np
import time
from pygame import mixer

# =========================
# CONFIG
# =========================
EAR_THRESHOLD = 0.25

# =========================
# COLORS
# =========================
GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

# =========================
# SOUND
# =========================
mixer.init()
mixer.music.load("alarm.wav")

# =========================
# MEDIAPIPE
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# =========================
# FUNCTION
# =========================
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(points, landmarks, w, h):
    coords = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in points]

    A = euclidean(coords[1], coords[5])
    B = euclidean(coords[2], coords[4])
    C = euclidean(coords[0], coords[3])

    return (A + B) / (2.0 * C)

# =========================
# MAIN
# =========================
def run_detector():
    cap = cv2.VideoCapture(0)

    # 🔥 PRO PERFORMANCE FIX
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow("Driver HUD", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Driver HUD", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    closed_start = None
    alarm_on = False
    running = True

    while running:

        # 🔥 PRO FIX: flush buffer (prevents lag frames)
        cap.grab()
        ret, frame = cap.retrieve()

        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = face_mesh.process(rgb)

        drowsy = False
        status = "SAFE DRIVE"
        status_color = GREEN
        avg_ear = 0

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark

            left_ear = calculate_ear(LEFT_EYE, landmarks, w, h)
            right_ear = calculate_ear(RIGHT_EYE, landmarks, w, h)

            avg_ear = (left_ear + right_ear) / 2.0

            # =========================
            # STRICT TIMING (1.7–2.4 sec)
            # =========================
            if avg_ear < EAR_THRESHOLD:
                if closed_start is None:
                    closed_start = time.time()

                elapsed = time.time() - closed_start

                if 1.7 <= elapsed <= 2.4 or elapsed > 2.4:
                    drowsy = True
            else:
                closed_start = None
                drowsy = False

        # =========================
        # ALARM CONTROL
        # =========================
        if drowsy and not alarm_on:
            mixer.music.play(-1)
            alarm_on = True

        elif not drowsy and alarm_on:
            mixer.music.stop()
            alarm_on = False

        # =========================
        # STATUS
        # =========================
        if drowsy:
            status = "⚠ DROWSINESS ALERT"
            status_color = RED

        # =========================
        # HUD OVERLAY
        # =========================
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.40, frame, 0.60, 0)

        cv2.rectangle(frame, (0, 0), (w, 80), status_color, -1)

        cv2.putText(frame,
                    status,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 0),
                    3)

        cv2.rectangle(frame,
                      (int(w*0.25), int(h*0.2)),
                      (int(w*0.75), int(h*0.8)),
                      WHITE,
                      1)

        if drowsy:
            alert = frame.copy()
            cv2.rectangle(alert, (0, 0), (w, h), RED, -1)
            frame = cv2.addWeighted(alert, 0.25, frame, 0.75, 0)

            cv2.putText(frame,
                        "TAKE A BREAK!",
                        (int(w/2)-200, int(h/2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        WHITE,
                        5)

        cv2.putText(frame,
                    f"EAR: {avg_ear:.2f}",
                    (30, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    WHITE,
                    2)

        cv2.imshow("Driver HUD", frame)

        # =========================
        # 🚀 INSTANT EXIT
        # =========================
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            running = False
            break

    # =========================
    # CLEAN EXIT (PRO LEVEL)
    # =========================
    try:
        cap.release()
    except:
        pass

    try:
        mixer.music.stop()
    except:
        pass

    cv2.destroyAllWindows()

    # 🔥 flush OpenCV event buffer
    for _ in range(5):
        cv2.waitKey(1)

run_detector()