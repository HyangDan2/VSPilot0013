from __future__ import annotations
from typing import Optional
import os, time
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QPixmap, QAction
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QHBoxLayout, QVBoxLayout,
    QComboBox, QSlider, QCheckBox, QGroupBox, QFormLayout,
    QPushButton, QFileDialog, QTextEdit, QMessageBox, QSpinBox
)

from src.utils import bgr_to_qimage
from src.processor import VideoProcessor
from src.detection import FaceMeshDetector, DrowsyMonitor
from src.kernel_editor import KernelEditorDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing & Ocular Detection Suite")
        self.setMinimumSize(QSize(1280, 720))

        # Runtime
        self.cap: Optional[cv2.VideoCapture] = None
        self.timer = QTimer(self); self.timer.timeout.connect(self.on_timer)
        self.mode_image = False
        self.loaded_image: Optional[np.ndarray] = None

        # Modules
        self.processor = VideoProcessor()
        self.detector = FaceMeshDetector()
        self.drowsy = DrowsyMonitor(frames_thresh=15)

        # Log file
        self.log_path = os.path.abspath("drowsy_log.txt")

        # UI build
        self._build_ui()
        self._open_camera(0)
        self.timer.start(30)
        self._log(f"▶ Session started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ---------- UI ----------
    def _build_ui(self):
        self.view_orig = QLabel("Original"); self._style_view(self.view_orig)
        self.view_proc = QLabel("Processed"); self._style_view(self.view_proc)

        view_box = QHBoxLayout()
        view_box.addWidget(self.view_orig, 1)
        view_box.addWidget(self.view_proc, 1)

        ctrl = self._build_controls()

        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(90)

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.addWidget(ctrl)
        lay.addLayout(view_box, 1)
        lay.addWidget(self.log)
        self.setCentralWidget(root)

        # Menu
        act_kernel = QAction("Custom Kernel Editor", self)
        act_kernel.triggered.connect(self.open_kernel_editor)
        menubar = self.menuBar()
        # Top-level menus
        file_menu  = menubar.addMenu("&File")
        tools_menu = menubar.addMenu("&Tools")
        help_menu  = menubar.addMenu("&Help")

        # Actions
        act_kernel = QAction("Custom Kernel Editor", self)
        act_kernel.triggered.connect(self.open_kernel_editor)

        # (선택) 표준 액션들 추가
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        # macOS에서 App 메뉴로 이동시키기 위한 역할 지정
        act_quit.setMenuRole(QAction.MenuRole.QuitRole)

        act_about = QAction("About", self)
        act_about.setMenuRole(QAction.MenuRole.AboutRole)
        act_about.triggered.connect(lambda: QMessageBox.information(
            self, "About",
            "Image Processing & Ocular Detection Suite\nPySide6 · OpenCV · MediaPipe"
        ))

        # 메뉴에 달기
        tools_menu.addAction(act_kernel)
        file_menu.addAction(act_quit)
        help_menu.addAction(act_about)

    def _style_view(self, lbl: QLabel):
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("background:#222;color:#ddd;")
        lbl.setMinimumSize(640, 360)

    def _build_controls(self) -> QWidget:
        box = QGroupBox("Controls")
        f = QFormLayout()

        # Source
        self.combo_source = QComboBox(); self.combo_source.addItems(["Webcam", "Image"])
        self.combo_source.currentIndexChanged.connect(self.on_source_changed)
        self.btn_load = QPushButton("Load Image"); self.btn_load.clicked.connect(self.on_load_image)

        self.combo_cam = QComboBox(); self.combo_cam.addItems([str(i) for i in range(0, 6)])
        self.combo_cam.currentIndexChanged.connect(self.on_camera_changed)

        # Color/CLAHE/Blur
        self.combo_color = QComboBox(); self.combo_color.addItems(["BGR", "Grayscale"])
        self.chk_clahe = QCheckBox("CLAHE")
        self.sld_clahe = QSlider(Qt.Orientation.Horizontal); self.sld_clahe.setRange(10,80); self.sld_clahe.setValue(20)

        self.combo_blur = QComboBox(); self.combo_blur.addItems(["None", "Gaussian", "Median", "Bilateral"])
        self.sld_blur = QSlider(Qt.Orientation.Horizontal); self.sld_blur.setRange(0,10); self.sld_blur.setValue(0)

        # Edge/Sharpen
        self.combo_edge = QComboBox(); self.combo_edge.addItems(["None", "Laplacian", "Sobel", "Canny", "Unsharp", "Custom Kernel"])

        # Intensity
        self.sld_intensity = QSlider(Qt.Orientation.Horizontal); self.sld_intensity.setRange(0,100); self.sld_intensity.setValue(100)

        # Detection / overlays
        self.chk_detect = QCheckBox("Detect on processed"); self.chk_detect.setChecked(True)
        self.chk_eye_outline = QCheckBox("Eye Outline"); self.chk_eye_outline.setChecked(True)
        self.chk_bbox = QCheckBox("BBox"); self.chk_bbox.setChecked(True)
        self.chk_label = QCheckBox("Label"); self.chk_label.setChecked(True)

        # Drowsy
        self.sld_ear = QSlider(Qt.Orientation.Horizontal); self.sld_ear.setRange(5,40); self.sld_ear.setValue(22)  # 0.05~0.40
        self.spn_ear_digits = QSpinBox(); self.spn_ear_digits.setRange(0,4); self.spn_ear_digits.setValue(2)
        self.chk_alarm = QCheckBox("Drowsy Alarm"); self.chk_alarm.setChecked(True)

        # Layout rows
        hsrc = self._hrow(self.combo_source, self.btn_load, QLabel("Camera"), self.combo_cam)
        f.addRow("Source", hsrc)
        f.addRow("Color", self.combo_color)
        f.addRow("CLAHE", self._hrow(self.chk_clahe, self.sld_clahe))
        f.addRow("Blur", self._hrow(self.combo_blur, self.sld_blur))
        f.addRow("Edge/Sharpen", self.combo_edge)
        f.addRow("Intensity", self.sld_intensity)
        f.addRow("Detection", self._hrow(self.chk_detect, self.chk_eye_outline, self.chk_bbox, self.chk_label))
        f.addRow("Drowsy", self._hrow(QLabel("EAR thresh"), self.sld_ear, QLabel("digits"), self.spn_ear_digits))
        f.addRow("", self.chk_alarm)

        box.setLayout(f)
        return box

    def _hrow(self, *widgets):
        w = QWidget()
        h = QHBoxLayout(w)
        for x in widgets:
            h.addWidget(x)
        return w

    # ---------- Source/Cam ----------
    def on_source_changed(self, _):
        self.mode_image = (self.combo_source.currentText() == "Image")
        if not self.mode_image and self.cap is None:
            self._open_camera(int(self.combo_cam.currentText()))

    def on_camera_changed(self, _):
        if self.mode_image: return
        self._open_camera(int(self.combo_cam.currentText()))

    def on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path: return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "Load failed", "이미지를 열 수 없어요.")
            return
        self.loaded_image = img.copy()
        self.mode_image = True
        self.combo_source.setCurrentText("Image")

    def _open_camera(self, index: int):
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera", f"카메라 {index} 열 수 없음.")
            self.cap = None; return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    # ---------- Timer loop ----------
    def on_timer(self):
        frame = None
        if self.mode_image:
            if self.loaded_image is None:
                self.view_orig.setText("Load an image…")
                self.view_proc.setText("Load an image…")
                return
            frame = self.loaded_image.copy()
        else:
            if not self.cap: return
            ok, frm = self.cap.read()
            if not ok: return
            frame = frm

        original = frame.copy()

        # sync processor with UI
        self.processor.color_mode = self.combo_color.currentText()
        self.processor.clahe_enabled = self.chk_clahe.isChecked()
        self.processor.clahe_clip = self.sld_clahe.value() / 10.0
        self.processor.blur_kind = self.combo_blur.currentText()
        self.processor.blur_level = self.sld_blur.value()
        self.processor.edge_kind = self.combo_edge.currentText()
        self.processor.intensity = self.sld_intensity.value() / 100.0

        processed = self.processor.process(frame.copy(), base_for_blend=original)
        out = processed.copy()

        # detection on processed (fallback on original for edge modes)
        if self.chk_detect.isChecked() and self.detector.mesh:
            results = self.detector.detect(processed)
            if (not results or not results.multi_face_landmarks) and \
               self.processor.edge_kind in ("Canny", "Laplacian", "Sobel"):
                results = self.detector.detect(original)

            out, ear_val, _ = self.detector.draw_overlays(
                results, out,
                draw_eye_outline=self.chk_eye_outline.isChecked(),
                draw_bbox=self.chk_bbox.isChecked(),
                draw_label=self.chk_label.isChecked(),
                ear_digits=self.spn_ear_digits.value()
            )

            if self.chk_alarm.isChecked():
                thresh = self.sld_ear.value() / 100.0
                turned_on, turned_off = self.drowsy.update(ear_val, thresh)
                if turned_on:
                    self._log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ALARM: Drowsy (EAR={ear_val:.3f} < {thresh:.3f})")
                if turned_off:
                    self._log("ALARM OFF (recovered)")

                if self.drowsy.is_active():
                    h, w = out.shape[:2]
                    cv2.rectangle(out, (0, 0), (w, 50), (0, 0, 255), -1)
                    cv2.putText(out, "DROWSY!", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

        # show
        self._set_pixmap(self.view_orig, original)
        self._set_pixmap(self.view_proc, out)

    # ---------- Kernel editor ----------
    def open_kernel_editor(self):
        dlg = KernelEditorDialog(self)
        def preview_cb(ker):
            # grab one frame for preview
            if self.mode_image and self.loaded_image is not None:
                base = self.loaded_image.copy()
            else:
                if not self.cap: return
                ok, frm = self.cap.read()
                if not ok: return
                base = frm
            prev = cv2.filter2D(base, -1, ker)
            self._set_pixmap(self.view_proc, prev)
        dlg.preview_callback = preview_cb
        dlg.kernel_applied.connect(self._apply_custom_kernel)
        dlg.exec()

    def _apply_custom_kernel(self, ker):
        self.processor.set_custom_kernel(ker)
        self.combo_edge.setCurrentText("Custom Kernel")

    # ---------- Helpers ----------
    def _set_pixmap(self, label: QLabel, img: np.ndarray):
        qimg = bgr_to_qimage(img)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(
            label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def _log(self, msg: str):
        self.log.append(msg)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    def closeEvent(self, e):
        self.timer.stop()
        if self.cap: self.cap.release()
        if self.detector: self.detector.close()
        self._log(f"■ Session ended at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        super().closeEvent(e)
