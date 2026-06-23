import pickle
import numpy as np
import cv2

try:
    with open("focus_model.pkl", "rb") as f:
        focus_model = pickle.load(f)
    MODEL_LOADED = True
except Exception:
    MODEL_LOADED = False

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def _ear(lm, idx, w, h):
    p = [(lm[i].x * w, lm[i].y * h) for i in idx]
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (A + B) / (2.0 * C + 1e-6)

class FocusDetector:
    def __init__(self):
        self.blink_count = 0
        self._prev_ear   = 0.3
        self._mesh_obj   = None

    def _get_mesh(self):
        if self._mesh_obj is None:
            try:
                import mediapipe as mp
                self._mesh_obj = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception:
                self._mesh_obj = None
        return self._mesh_obj

    def detect(self, frame):
        if not MODEL_LOADED:
            return {"focus_score": 50, "state": "Moderate",
                    "prediction": 1, "blinks": self.blink_count}

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            mesh = self._get_mesh()
            if mesh is None:
                return {"focus_score": 50, "state": "Moderate",
                        "prediction": 1, "blinks": self.blink_count}
            res = mesh.process(rgb)
        except Exception:
            return {"focus_score": 50, "state": "Moderate",
                    "prediction": 1, "blinks": self.blink_count}

        if not res.multi_face_landmarks:
            return {"focus_score": 0, "state": "No face",
                    "prediction": 0, "blinks": self.blink_count}

        lm        = res.multi_face_landmarks[0].landmark
        le  = _ear(lm, LEFT_EYE,  w, h)
        re  = _ear(lm, RIGHT_EYE, w, h)
        avg = (le + re) / 2.0

        lw = np.linalg.norm(
            np.array([lm[LEFT_EYE[0]].x*w,  lm[LEFT_EYE[0]].y*h]) -
            np.array([lm[LEFT_EYE[3]].x*w,  lm[LEFT_EYE[3]].y*h]))
        rw = np.linalg.norm(
            np.array([lm[RIGHT_EYE[0]].x*w, lm[RIGHT_EYE[0]].y*h]) -
            np.array([lm[RIGHT_EYE[3]].x*w, lm[RIGHT_EYE[3]].y*h]))
        ed = np.linalg.norm(
            np.array([lm[33].x*w, lm[33].y*h]) -
            np.array([lm[362].x*w, lm[362].y*h]))

        features = np.array([[le, re, avg, lw, rw, ed]])

        if self._prev_ear > 0.25 and avg < 0.20:
            self.blink_count += 1
        self._prev_ear = avg

        try:
            pred  = int(focus_model.predict(features)[0])
            proba = focus_model.predict_proba(features)[0]
            score = int(proba[1] * 100)
        except Exception:
            pred  = 1
            score = 50

        return {
            "focus_score": score,
            "state":       "Focused" if pred == 1 else "Distracted",
            "prediction":  pred,
            "blinks":      self.blink_count
        }
