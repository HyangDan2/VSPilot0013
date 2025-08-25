from __future__ import annotations
from typing import Optional
import cv2
import numpy as np
from src.utils import ensure_odd

class VideoProcessor:
    """
    전처리/후처리 파이프라인.
    - color_mode: "BGR" | "Grayscale"
    - clahe_enabled + clahe_clip
    - blur_kind: "None" | "Gaussian" | "Median" | "Bilateral"
    - blur_level: 0..10
    - edge_kind: "None" | "Laplacian" | "Sobel" | "Canny" | "Unsharp" | "Custom Kernel"
    - intensity: 0..1 (base와 처리본 가중합)
    - custom_kernel: Optional[np.ndarray]
    """
    def __init__(self):
        self.color_mode = "BGR"
        self.clahe_enabled = False
        self.clahe_clip = 2.0
        self.blur_kind = "None"
        self.blur_level = 0
        self.edge_kind = "None"
        self.intensity = 1.0
        self.custom_kernel: Optional[np.ndarray] = None

    def set_custom_kernel(self, kernel: Optional[np.ndarray]):
        self.custom_kernel = kernel

    def process(self, img: np.ndarray, base_for_blend: Optional[np.ndarray] = None) -> np.ndarray:
        # 1) Color
        if self.color_mode == "Grayscale":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2) CLAHE
        if self.clahe_enabled:
            clip = max(0.1, float(self.clahe_clip))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            if img.ndim == 2:
                img = clahe.apply(img)
            else:
                ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                ycc[:, :, 0] = clahe.apply(ycc[:, :, 0])
                img = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

        # 3) Blur
        if self.blur_kind != "None" and self.blur_level > 0:
            if self.blur_kind == "Gaussian":
                k = ensure_odd(2 * self.blur_level + 1)
                img = cv2.GaussianBlur(img, (k, k), self.blur_level * 2 + 1)
            elif self.blur_kind == "Median":
                k = ensure_odd(2 * self.blur_level + 1)
                img = cv2.medianBlur(img, k)
            elif self.blur_kind == "Bilateral":
                d = 5 + 2 * self.blur_level
                s = 10 + 10 * self.blur_level
                img = cv2.bilateralFilter(img, d, s, s)

        # 4) Edge / Sharpen / Canny / Unsharp / Custom
        if self.edge_kind != "None":
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.edge_kind == "Laplacian":
                lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
                lap = cv2.convertScaleAbs(lap)
                img = lap if img.ndim == 2 else cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

            elif self.edge_kind == "Sobel":
                sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                mag = cv2.magnitude(sx, sy)
                mag = (np.clip(mag / (mag.max() + 1e-6) * 255, 0, 255)).astype(np.uint8)
                img = mag if img.ndim == 2 else cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)

            elif self.edge_kind == "Canny":
                t = int(self.intensity * 100)  # 0~100
                lo = max(0, int(20 + 1.2 * t))
                hi = max(lo + 1, int(40 + 2.0 * t))
                edges = cv2.Canny(gray, lo, hi)
                img = edges if img.ndim == 2 else cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            elif self.edge_kind == "Unsharp":
                amount = float(self.intensity)  # 0~1
                blur = cv2.GaussianBlur(img, (3, 3), 0)
                img = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)

            elif self.edge_kind == "Custom Kernel" and self.custom_kernel is not None:
                img = cv2.filter2D(img, -1, self.custom_kernel)

        # 5) Intensity blend
        alpha = float(self.intensity)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if base_for_blend is not None:
            base = base_for_blend
            if base.shape[:2] != img.shape[:2]:
                base = cv2.resize(base, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            return cv2.addWeighted(base, 1.0 - alpha, img, alpha, 0.0)
        return img
