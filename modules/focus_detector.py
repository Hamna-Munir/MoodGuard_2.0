import pickle, numpy as np, cv2

try:
    with open("focus_model.pkl","rb") as f: focus_model=pickle.load(f)
    _MODEL=True
except: _MODEL=False

_L=[33,160,158,133,153,144]; _R=[362,385,387,263,373,380]

def _e(lm,idx,w,h):
    p=[(lm[i].x*w,lm[i].y*h) for i in idx]
    return (np.linalg.norm(np.array(p[1])-np.array(p[5]))+np.linalg.norm(np.array(p[2])-np.array(p[4])))/(2*np.linalg.norm(np.array(p[0])-np.array(p[3]))+1e-6)

class FocusDetector:
    def __init__(self):
        self.blink_count=0; self._pe=0.3; self._m=None
    def _mesh(self):
        if not self._m:
            try:
                import mediapipe as mp
                self._m=mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)
            except: pass
        return self._m
    def detect(self,frame):
        d={"focus_score":50,"state":"Moderate","prediction":1,"blinks":self.blink_count}
        if not _MODEL: return d
        h,w=frame.shape[:2]
        try:
            m=self._mesh()
            if not m: return d
            res=m.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        except: return d
        if not res.multi_face_landmarks:
            return {"focus_score":0,"state":"No face","prediction":0,"blinks":self.blink_count}
        lm=res.multi_face_landmarks[0].landmark
        le=_e(lm,_L,w,h); re=_e(lm,_R,w,h); avg=(le+re)/2
        lw=np.linalg.norm(np.array([lm[_L[0]].x*w,lm[_L[0]].y*h])-np.array([lm[_L[3]].x*w,lm[_L[3]].y*h]))
        rw=np.linalg.norm(np.array([lm[_R[0]].x*w,lm[_R[0]].y*h])-np.array([lm[_R[3]].x*w,lm[_R[3]].y*h]))
        ed=np.linalg.norm(np.array([lm[33].x*w,lm[33].y*h])-np.array([lm[362].x*w,lm[362].y*h]))
        if self._pe>0.25 and avg<0.20: self.blink_count+=1
        self._pe=avg
        try:
            pred=int(focus_model.predict([[le,re,avg,lw,rw,ed]])[0])
            score=int(focus_model.predict_proba([[le,re,avg,lw,rw,ed]])[0][1]*100)
        except: pred=1; score=50
        return {"focus_score":score,"state":"Focused" if pred else "Distracted","prediction":pred,"blinks":self.blink_count}
