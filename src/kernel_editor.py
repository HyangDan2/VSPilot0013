from __future__ import annotations
from typing import List, Optional
import numpy as np
import cv2
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QPlainTextEdit, QCheckBox, QHBoxLayout,
    QPushButton, QDialogButtonBox, QMessageBox
)

class KernelEditorDialog(QDialog):
    kernel_applied = Signal(object)  # np.ndarray or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Kernel Editor")
        self.setMinimumSize(500, 380)

        self.text = QPlainTextEdit()
        self.text.setPlaceholderText(
            "예시:\n0 -1 0\n-1 5 -1\n0 -1 0\n\n쉼표 또는 공백으로 구분"
        )
        self.chk_norm = QCheckBox("합이 0이 아니면 1로 정규화(합=1)")
        self.chk_norm.setChecked(False)

        self.btn_preview = QPushButton("미리보기(현재 프레임)")
        self.btn_clear = QPushButton("비우기")
        self.btn_preview.clicked.connect(self.on_preview)
        self.btn_clear.clicked.connect(lambda: self.text.setPlainText(""))

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox.accepted.connect(self.on_ok)
        self.buttonBox.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(self.text)
        lay.addWidget(self.chk_norm)
        h = QHBoxLayout()
        h.addWidget(self.btn_preview)
        h.addWidget(self.btn_clear)
        lay.addLayout(h)
        lay.addWidget(self.buttonBox)

        self.preview_callback = None  # set by parent

    def parse_kernel(self) -> Optional[np.ndarray]:
        s = self.text.toPlainText().strip()
        if not s:
            return None
        rows = [r for r in s.splitlines() if r.strip()]
        data: List[List[float]] = []
        for r in rows:
            parts = r.replace(",", " ").split()
            data.append([float(x) for x in parts])
        widths = {len(r) for r in data}
        if len(widths) != 1:
            raise ValueError("모든 행의 열 개수가 같아야 합니다.")
        ker = np.array(data, dtype=np.float32)
        if self.chk_norm.isChecked():
            sm = ker.sum()
            if abs(sm) > 1e-6:
                ker = ker / sm
        return ker

    def on_preview(self):
        try:
            ker = self.parse_kernel()
        except Exception as e:
            QMessageBox.warning(self, "파싱 오류", str(e))
            return
        if ker is None:
            QMessageBox.information(self, "안내", "커널이 비어 있어요.")
            return
        if self.preview_callback:
            self.preview_callback(ker)

    def on_ok(self):
        try:
            ker = self.parse_kernel()
        except Exception as e:
            QMessageBox.warning(self, "파싱 오류", str(e))
            return
        self.kernel_applied.emit(ker)
        self.accept()
