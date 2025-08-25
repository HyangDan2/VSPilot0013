# Image Processing & Ocular Detection Suite
A PySide6 · OpenCV · MediaPipe based desktop application for **real-time image processing and ocular detection**.

This application provides a **real-time video/image processing pipeline** together with **ocular (eye) landmark detection, EAR calculation, and drowsiness alarm**.  
Both the original and processed images are displayed **side by side**, with support for **Webcam/Image source selection**, **camera index switching**, and advanced filters including **Canny, Unsharp Mask, and Custom Kernel Editor**.

---

## ✨ Features
- **Dual View**: Original and Processed view (aspect ratio preserved)
- **Source selection**: Webcam / Image
- **Camera index selection**: switch between cameras (0–5)
- **Preprocessing pipeline**
  - Color: BGR / Grayscale
  - CLAHE (clipLimit adjustable)
  - Blur: Gaussian / Median / Bilateral (+intensity control)
  - Edge/Sharpen: Laplacian / Sobel / Canny / Unsharp / Custom Kernel
  - Intensity: blending between original and processed image
- **Eye detection & drowsiness monitoring**
  - MediaPipe FaceMesh based EAR calculation (average of both eyes)
  - Eye outline, bounding box, label (EAR) toggle
  - EAR threshold slider (0.05–0.40)
  - Frame-based threshold for **Drowsy Alarm** + red banner + logging to file (`drowsy_log.txt`)
- **Custom Kernel Editor**
  - Enter arbitrary kernel matrices (comma/space separated)
  - Optional normalization (sum=1)
  - Preview on current frame
  - Apply immediately

---

## 🧰 Installation
```bash
pip install -r requirements.txt
```
▶ Run
```bash
python -m src.app
```

## 📂 Project Structure
```pgsql            
src/
  app.py             # entry point
  main_window.py     # main UI and orchestration
  processor.py       # image processing pipeline
  detection.py       # FaceMesh + EAR detection + drowsiness monitor
  kernel_editor.py   # custom kernel editor dialog
  utils.py           # helper utilities (QImage conversion, etc.)
```

## ⚙️ Tech Overview
- **PySide6**: GUI framework (QMainWindow, QWidget, QLabel, QSlider, QComboBox, etc.)
- **OpenCV**: Image/video processing (filters, edges, blur, kernel operations)
- **MediaPipe FaceMesh**: 468 facial landmarks, 6-point EAR calculation
- **Modular OOP design**:
  - VideoProcessor: Preprocessing pipeline
  - FaceMeshDetector: FaceMesh inference + EAR overlays
  - DrowsyMonitor: EAR-based state machine
  - KernelEditorDialog: Kernel input and preview
  - MainWindow: Main GUI and orchestration

## 🧪 Tips & Troubleshooting
Detection failures: edge-heavy outputs (Canny/Laplacian/Sobel) may reduce detection accuracy. → Fallback to original frame is automatically applied.

- Low FPS: for high resolution, consider moving capture/processing/rendering to QThreads.
- EAR threshold: default = 0.22, but adjust depending on lighting, distance, and camera.

## 📜 License
This project is licensed under the MIT License. See LICENSE.txt for details.

---

## 📋 `requirements.txt`

```text
PySide6>=6.6
opencv-python>=4.8
mediapipe>=0.10
numpy>=1.24
```