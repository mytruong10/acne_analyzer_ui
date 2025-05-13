Acne Analysis and Grading System

Overview:
This project provides an automated system for acne detection and grading using deep learning. It combines a YOLO-based object detection model for acne localization and a custom STN-ResNet18 classification model for severity grading. The system features a PyQt5-based GUI for real-time webcam capture, image analysis, and result visualization.

Features:
- Webcam Integration: Capture live images directly from your webcam.
- Acne Detection: Uses a YOLO model to detect acne regions and draw bounding boxes.
- Severity Grading: Classifies acne severity into four levels (Mild, Moderate, Severe, Very Severe) using a spatial transformer network (STN) with ResNet18.
- Result Visualization: Displays detected regions and grading results in a user-friendly GUI.

Requirements:
- Python 3.7+
- PyTorch
- torchvision
- ultralytics (YOLO)
- OpenCV
- PyQt5
- PIL (Pillow)

Install dependencies with:
pip install torch torchvision ultralytics opencv-python pyqt5 pillow numpy

