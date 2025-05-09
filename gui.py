from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)

        # Create a stacked widget to switch between pages
        self.stackedWidget = QtWidgets.QStackedWidget(MainWindow)
        self.stackedWidget.setGeometry(QtCore.QRect(0, 0, 800, 600))
        self.stackedWidget.setObjectName("stackedWidget")

        # Create the first page (camera feed and buttons)
        self.page1 = QtWidgets.QWidget()
        self.page1.setObjectName("page1")

        # Add the widgets to page1
        self.label = QtWidgets.QLabel(self.page1)
        self.label.setGeometry(QtCore.QRect(30, 50, 741, 431))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setLineWidth(3)
        self.label.setText("")
        self.label.setObjectName("label")

        # Create a horizontal layout for buttons
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.page1)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(30, 500, 741, 41))  
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")

        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)

        # Create buttons
        self.Button_start = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.Button_start.setObjectName("Button_start")
        self.horizontalLayout.addWidget(self.Button_start)

        self.Button_capture = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.Button_capture.setObjectName("Button_capture")
        self.horizontalLayout.addWidget(self.Button_capture)

        self.Button_analyze = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.Button_analyze.setObjectName("Button_analyze")
        self.horizontalLayout.addWidget(self.Button_analyze)

        # Add the first page to the stacked widget
        self.stackedWidget.addWidget(self.page1)

        # Create the second page for displaying the grading result and detected image
        self.page2 = QtWidgets.QWidget()
        self.page2.setObjectName("page2")

        # QLabel to display the image with bounding boxes from YOLO model
        self.result_label = QtWidgets.QLabel(self.page2)
        self.result_label.setGeometry(QtCore.QRect(30, 50, 741, 431))
        self.result_label.setFrameShape(QtWidgets.QFrame.Box)
        self.result_label.setLineWidth(3)
        self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_label.setObjectName("result_label")

        # QLabel to display the grading result (text)
        self.grading_result = QtWidgets.QLabel(self.page2)
        self.grading_result.setGeometry(QtCore.QRect(30, 500, 741, 25))
        self.grading_result.setAlignment(QtCore.Qt.AlignCenter)
        self.grading_result.setObjectName("grading_result")

        self.Button_back = QtWidgets.QPushButton(self.page2)
        self.Button_back.setGeometry(QtCore.QRect(320, 530, 161, 25))
        self.Button_back.setObjectName("Button_back")

        # Add the second page to the stacked widget
        self.stackedWidget.addWidget(self.page2)

        # Set the first page as the initial view
        self.stackedWidget.setCurrentWidget(self.page1)

        # Add the stacked widget to the main window
        MainWindow.setCentralWidget(self.stackedWidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Button_start.setText(_translate("MainWindow", "Turn on webcam"))
        self.Button_capture.setText(_translate("MainWindow", "Capture"))
        self.Button_analyze.setText(_translate("MainWindow", "Analyze"))
        self.Button_back.setText(_translate("MainWindow", "Back to camera"))
        self.grading_result.setText(_translate("MainWindow", "Grading result will be displayed here."))
