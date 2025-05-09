import sys
import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from gui import Ui_MainWindow
import grading_module  # Import your grading module


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)

        # Connect the buttons
        self.uic.Button_start.clicked.connect(self.start_capture_video)
        self.uic.Button_capture.clicked.connect(self.capture_image)
        self.uic.Button_analyze.clicked.connect(self.analyze_image)
        self.uic.Button_back.clicked.connect(self.switch_to_camera_page)

        # Initialize variables
        self.thread = None
        self.current_frame = None

    def closeEvent(self, event):
        """Stop video capture gracefully when the window is closed."""
        if self.thread is not None:
            self.thread.stop()

    def start_capture_video(self):
        """Start capturing video from the webcam."""
        if self.thread is None or not self.thread.isRunning():
            self.thread = CaptureVideoThread()
            self.thread.start()
            self.thread.signal.connect(self.show_webcam)

    def show_webcam(self, cv_img):
        """Updates the image_label with a new OpenCV image."""
        self.current_frame = cv_img  # Store the current frame
        qt_img = self.convert_cv_qt(cv_img)
        self.uic.label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an OpenCV image to QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def capture_image(self):
        """Save the current frame to a file."""
        if self.current_frame is not None:
            cv2.imwrite("captured_image.jpg", self.current_frame)
            print("Image saved as 'captured_image.jpg'")

    def analyze_image(self):
        """Analyze the captured image using the grading module, draw bounding boxes, and display the result."""
        if self.current_frame is not None:
            # Save the current frame to a temporary file
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, self.current_frame)
            
            # Call the YOLO model to detect acne and draw bounding boxes
            results = grading_module.yolo_model(image_path)  
            detected_img = self.draw_bounding_boxes(self.current_frame, results)  

            # Call the grading module to analyze the image
            grading = grading_module.grade_acne(image_path, grading_module.yolo_model, grading_module.classification_model, grading_module.device)

            # Convert the image with bounding boxes to QPixmap and display it on the second page
            qt_img = self.convert_cv_qt(detected_img)
            self.uic.result_label.setPixmap(qt_img)

            # Set the grading result text
            self.uic.grading_result.setText(f"Acne Grading: {grading}")

            # Switch to the second page
            self.uic.stackedWidget.setCurrentWidget(self.uic.page2)

    def draw_bounding_boxes(self, image, results):
        """Draw bounding boxes on the image based on YOLO results."""
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Draw the bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add labels for the detected acne
                label = f"{box.conf[0]:.2f}"
                font_scale = 0.5  
                thickness = 1  
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10  # Adjust position to stay within the image
                cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        return image

    def switch_to_camera_page(self):
        """Switch back to the camera page."""
        self.uic.stackedWidget.setCurrentWidget(self.uic.page1)


class CaptureVideoThread(QThread):
    """Thread for capturing video from the webcam."""
    signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super(CaptureVideoThread, self).__init__()
        self._is_running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._is_running:
            ret, cv_img = cap.read()
            if ret:
                self.signal.emit(cv_img)

    def stop(self):
        """Stop the video capture thread."""
        self._is_running = False
        self.quit()
        self.wait()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
