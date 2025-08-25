from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
import math

try:
    import mediapipe as mp
    MP_OK = True
except Exception:
    MP_OK = False

# 6포인트 EAR 계산 (dlib 방식)
LEFT_EYE_6  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_6 = [263, 387, 385, 362, 380, 373]

def _dist(p, q):
    return math.hypot(p[0]-q[0], p[1]-q[1])

def eye_aspect_ratio(pts6):
    p1, p2, p3, p4, p5, p6 = pts6
    num = _dist(p2, p6) + _dist(p3, p5)
    den = 2.0 * _dist(p1, p4)
    return (num / den) if den > 1e-6 else 0.0

@dataclass
class DrowsyState:
    below_counter: int = 0
    active: bool = False

class DrowsyMonitor:
    """EAR 기반 졸림 감지 상태 관리."""
    def __init__(self, frames_thresh: int = 15):
        self.state = DrowsyState()
        self.frames_thresh = frames_thresh

    def update(self, ear: float, thresh: float) -> tuple[bool, bool]:
        """
        Returns (alarm_on_now, alarm_off_now).
        """
        turned_on = False
        turned_off = False

        if ear > 0 and ear < thresh:
            self.state.below_counter += 1
        else:
            if self.state.active:
                turned_off = True
            self.state.below_counter = 0
            self.state.active = False

        if not self.state.active and self.state.below_counter >= self.frames_thresh:
            self.state.active = True
            turned_on = True

        return turned_on, turned_off

    def is_active(self) -> bool:
        return self.state.active

class FaceMeshDetector:
    """MediaPipe FaceMesh로 EAR 계산 & 오버레이 생성."""
    def __init__(self):
        self.mesh = None
        if MP_OK:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def close(self):
        if self.mesh:
            self.mesh.close()

    def detect(self, bgr) -> Optional[object]:
        """bgr 프레임에서 landmarks 탐지 결과 반환."""
        if not self.mesh:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.mesh.process(rgb)

    def draw_overlays(
        self,
        results,
        canvas: np.ndarray,
        draw_eye_outline: bool = True,
        draw_bbox: bool = True,
        draw_label: bool = True,
        ear_digits: int = 2,
    ) -> Tuple[np.ndarray, float, Optional[Tuple[int,int,int,int]]]:
        """이미 계산된 results로 캔버스에 그리기 + EAR 반환."""
        h, w = canvas.shape[:2]
        ear_val = 0.0
        bbox = None

        if not (results and results.multi_face_landmarks):
            return canvas, ear_val, bbox

        lms = results.multi_face_landmarks[0].landmark

        def px(idx):
            return (int(lms[idx].x * w), int(lms[idx].y * h))

        L = [px(i) for i in LEFT_EYE_6]
        R = [px(i) for i in RIGHT_EYE_6]
        ear_left  = eye_aspect_ratio(L)
        ear_right = eye_aspect_ratio(R)
        ear_val = (ear_left + ear_right) / 2.0

        if draw_eye_outline:
            ptsL = np.array(L + [L[0]], np.int32).reshape(-1, 1, 2)
            ptsR = np.array(R + [R[0]], np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [ptsL], False, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.polylines(canvas, [ptsR], False, (255, 0, 0), 2, cv2.LINE_AA)

        if draw_bbox:
            xs = [int(l.x * w) for l in lms]
            ys = [int(l.y * h) for l in lms]
            x1, y1 = max(0, min(xs)), max(0, min(ys))
            x2, y2 = min(w - 1, max(xs)), min(h - 1, max(ys))
            bbox = (x1, y1, x2, y2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if draw_label:
            cv2.putText(canvas, f"EAR {ear_val:.{ear_digits}f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        return canvas, ear_val, bbox
