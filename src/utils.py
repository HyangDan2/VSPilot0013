from __future__ import annotations
import cv2
import numpy as np
from PySide6.QtGui import QImage

def bgr_to_qimage(img: np.ndarray) -> QImage:
    """Convert BGR (or GRAY) numpy array to QImage."""
    if img.ndim == 2:
        h, w = img.shape
        return QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888).copy()

def ensure_odd(k: int) -> int:
    """Force an odd kernel size (>=1)."""
    return k if k % 2 == 1 else max(1, k - 1)
