import cv2
import mediapipe as mp
import numpy as np
import time
from pygame import mixer

EAR_LIMIT = 0.25  

# COLORS
GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

# SOUND
mixer.init()
mixer.music.load("alarm.wav")

# MEDIAPIPE
mp_face = mp.solutions.face_mesh       
face = mp_face.FaceMesh(refine_landmarks=True) 

LEFT = [33, 160, 158, 133, 153, 144]  
RIGHT = [362, 385, 387, 263, 373, 380] 

# FUNCTION
def dist(p1, p2):   
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_ear(points, lm, w, h): 
    pts = [(int(lm[p].x * w), int(lm[p].y * h)) for p in points]

    A = dist(pts[1], pts[5])
    B = dist(pts[2], pts[4])
    C = dist(pts[0], pts[3])

    return (A+B)/(2.0*C)

# MAIN
def run():
    cam = cv2.VideoCapture(0)  
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow("Driver HUD", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Driver HUD", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start_time = None    
    alarm = False      
    running = True

    while running:
        cam.grab()
        ret, frame = cam.retrieve()

        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = face.process(rgb)

        sleepy = False        
        status = "SAFE DRIVE"
        color = GREEN
        ear = 0             

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark

            left = get_ear(LEFT, lm, w, h)
            right = get_ear(RIGHT, lm, w, h)

            ear = (left + right) / 2.0

            if ear < EAR_LIMIT:
                if start_time is None:
                    start_time = time.time()

                t = time.time() - start_time

                if t >= 1.7:   
                    sleepy = True
            else:
                start_time = None
                sleepy = False

        # ALARM
        if sleepy and not alarm:
            mixer.music.play(-1)
            alarm = True

        elif not sleepy and alarm:
            mixer.music.stop()
            alarm = False

        # STATUS
        if sleepy:
            status = "DROWSINESS ALERT"
            color = RED

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.40, frame, 0.60, 0)

        cv2.rectangle(frame, (0, 0), (w, 80), color, -1)

        cv2.putText(frame,status,(30, 50),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0, 0, 0),3)

        cv2.rectangle(frame,(int(w*0.25), int(h*0.2)),(int(w*0.75), int(h*0.8)),WHITE, 1)

        if sleepy:
            alert = frame.copy()
            cv2.rectangle(alert, (0, 0), (w, h), RED, -1)
            frame = cv2.addWeighted(alert, 0.25, frame, 0.75, 0)

            cv2.putText(frame,"TAKE A BREAK!",(int(w/2)-200, int(h/2)),cv2.FONT_HERSHEY_SIMPLEX,2,WHITE,5)

        cv2.putText(frame,f"EAR: {ear:.2f}",(30, h - 40),cv2.FONT_HERSHEY_SIMPLEX,0.8,WHITE,2)
        cv2.imshow("Driver HUD", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            running = False
            break

    try:
        cam.release()
    except:
        pass

    try:
        mixer.music.stop()
    except:
        pass

    cv2.destroyAllWindows()

    for _ in range(5):
        cv2.waitKey(1)

run()