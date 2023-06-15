from PyQt5 import QtCore, QtGui, QtWidgets
import tkinter as tk
from tkinter import filedialog
from controller.Controller import ControllerClass
from helpers.helpers import TrainerANNData, Trainer3DCNNData, TrainerCNNData
from utils.constants import (
    WIDTH_3D_CNN,
    HEIGHT_3D_CNN, 
    WIDTH_PICTURE_CNN, 
    HEIGHT_PICTURE_CNN,
    DETECTION_CONFIDENCE_ANN,
    TRACKING_CONFIDENCE_ANN,
    COMPLEXITY_ANN,
)
from math import floor
import cv2


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(854, 683)
        MainWindow.setMinimumSize(QtCore.QSize(854, 683))
        MainWindow.setMaximumSize(QtCore.QSize(854, 683))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(10, 10, 831, 61))
        self.textBrowser.setObjectName("textBrowser")
        self.insert_button = QtWidgets.QPushButton(self.centralwidget)
        self.insert_button.setGeometry(QtCore.QRect(10, 280, 831, 71))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.insert_button.setFont(font)
        self.insert_button.setIconSize(QtCore.QSize(32, 32))
        self.insert_button.setObjectName("insert_button")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(10, 80, 341, 192))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.textBrowser_3 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_3.setGeometry(QtCore.QRect(10, 360, 171, 41))
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.textBrowser_4 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_4.setGeometry(QtCore.QRect(10, 420, 171, 41))
        self.textBrowser_4.setObjectName("textBrowser_4")
        self.textBrowser_5 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_5.setGeometry(QtCore.QRect(10, 480, 171, 41))
        self.textBrowser_5.setObjectName("textBrowser_5")
        self.ann_result = QtWidgets.QTextBrowser(self.centralwidget)
        self.ann_result.setGeometry(QtCore.QRect(190, 360, 221, 41))
        self.ann_result.setObjectName("ann_result")
        self.cnn_3d_result = QtWidgets.QTextBrowser(self.centralwidget)
        self.cnn_3d_result.setGeometry(QtCore.QRect(190, 480, 221, 41))
        self.cnn_3d_result.setObjectName("cnn_3d_result")
        self.cnn_result = QtWidgets.QTextBrowser(self.centralwidget)
        self.cnn_result.setGeometry(QtCore.QRect(190, 420, 221, 41))
        self.cnn_result.setObjectName("cnn_result")
        self.openGLWidget = QtWidgets.QLabel(self.centralwidget)
        self.openGLWidget.setGeometry(QtCore.QRect(420, 360, 421, 221))
        self.openGLWidget.setObjectName("openGLWidget")
        self.textBrowser_9 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_9.setGeometry(QtCore.QRect(10, 540, 171, 41))
        self.textBrowser_9.setObjectName("textBrowser_9")
        self.average_score = QtWidgets.QTextBrowser(self.centralwidget)
        self.average_score.setGeometry(QtCore.QRect(190, 540, 221, 41))
        self.average_score.setObjectName("average_score")
        self.final_message = QtWidgets.QTextBrowser(self.centralwidget)
        self.final_message.setGeometry(QtCore.QRect(10, 590, 831, 71))
        self.final_message.setObjectName("final_message")
        self.calendarWidget = QtWidgets.QCalendarWidget(self.centralwidget)
        self.calendarWidget.setGeometry(QtCore.QRect(370, 80, 471, 191))
        self.calendarWidget.setObjectName("calendarWidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.insert_button.clicked.connect(self.insert_video)
        self.root = tk.Tk()
        self.root.withdraw()

        self.trainer_ann_data = TrainerANNData(
            detection_confidece=DETECTION_CONFIDENCE_ANN,
            tracking_confidence=TRACKING_CONFIDENCE_ANN,
            complexity=COMPLEXITY_ANN,
        )

        self.trainer_3d_cnn_data = Trainer3DCNNData(
            width_3d_cnn = WIDTH_3D_CNN,
            height_3d_cnn = HEIGHT_3D_CNN,
        )

        self.trainer_cnn_data = TrainerCNNData(
            width_cnn=WIDTH_PICTURE_CNN,
            height_cnn=HEIGHT_PICTURE_CNN,
        )

        self.controller = ControllerClass(
            ann_data=self.trainer_ann_data,
            cnn_3d_data=self.trainer_3d_cnn_data,
            cnn_data=self.trainer_cnn_data,
        )



    def insert_video(self):
        file_path = filedialog.askopenfilename()
        extension = file_path[-3:]
        if extension in ("png", "jpg"):
            controller_cnn_str, controller_cnn_list = self.controller.evaluate_cnn_image(file_path)
            max_val_cnn = max(controller_cnn_list)
            max_val_cnn = max_val_cnn * (-1) if max_val_cnn  < 0 else max_val_cnn
            max_val_cnn = float(int(max_val_cnn * 10000) / 100.0)

            self.ann_result.setText("")
            self.cnn_3d_result.setText("")
            self.cnn_result.setText(f"{controller_cnn_str} with {max_val_cnn} %")

            self.average_score.setText(f"{max_val_cnn} %")

            if max_val_cnn >= 90.0:
                self.final_message.setText("Congratulation! Your form in the exercise is awesome")
            elif max_val_cnn >= 60 and max_val_cnn <= 89.9:
                self.final_message.setText("You form could do some work! But you are on the right track!")
            else:
                base_message = " Stop what you are doing. You need to fix your form immediately!\n Here is a video to help you out:\n"
                link_video = ""
                match controller_cnn_str:
                    case "ABS":
                        link_video = "https://www.youtube.com/watch?v=k1cLRd7cahQ&ab_channel=MikeThurston"
                    case "BACK":
                        link_video = "https://www.youtube.com/watch?v=2tnATDflg4o&ab_channel=JeremyEthier"
                    case "BICEPS":
                        link_video = "https://www.youtube.com/watch?v=TaeMP1WJTKw&ab_channel=GravityTransformation-FatLossExperts"
                    case "BUTT":
                        link_video = "https://www.youtube.com/watch?v=6vlP9xPJbaQ&ab_channel=KrissyCela"
                    case "CHEST":
                        link_video = "https://www.youtube.com/watch?v=89e518dl4I8&ab_channel=ATHLEAN-X%E2%84%A2"
                    case "FOREARM":
                        link_video = "https://www.youtube.com/watch?v=zVJ3MN4-kkQ&ab_channel=GravityTransformation-FatLossExperts"
                    case "LEGS":
                        link_video = "https://www.youtube.com/watch?v=uYkpTWfpFHA&t=1s&ab_channel=WORKOUT"
                    case "SHOULDER":
                        link_video = "https://www.youtube.com/watch?v=rnDWTXYDMOg&ab_channel=GravityTransformation-FatLossExperts"
                    case "TRICEPS":
                        link_video = "https://www.youtube.com/watch?v=SuajkDYlIRw&ab_channel=VShred"    
                self.final_message.setText(f"{base_message} {link_video} ")


            image = cv2.imread(file_path)
            image = cv2.resize(image, (421, 221))
            cv2.imwrite("./img/image.png", image)
            image = QtGui.QPixmap("./img/image.png")
            self.openGLWidget.setPixmap(image)
        else:
            controller_ann_str, controller_ann_list = self.controller.evaluate_ann_video(file_path)
            controller_3d_cnn_str, controller_3d_cnn_list = self.controller.evaluate_3d_cnn_video(file_path)

            cap = cv2.VideoCapture(file_path)
            count = 0

            image = ""
            while cap.isOpened():
                ret, frame = cap.read()

                if ret:
                    if count == 100:
                        image = frame
                        cap.release()
                        break
                    else:
                        count += 1
                else:
                    cap.release()
                    break
            
            image = cv2.resize(image, (421, 221))
            cv2.imwrite("./img/image.png", image)
            image = QtGui.QPixmap("./img/image.png")
            self.openGLWidget.setPixmap(image)

            max_val_ann = max(controller_ann_list)
            max_val_ann = max_val_ann * (-1) if max_val_ann  < 0 else max_val_ann
            max_val_ann = float(int(max_val_ann * 10000) / 100.0)
            max_val_3d_cnn = max(controller_3d_cnn_list)
            max_val_3d_cnn = max_val_3d_cnn * (-1) if max_val_3d_cnn  < 0 else max_val_3d_cnn
            max_val_3d_cnn = float(int(max_val_3d_cnn * 10000) / 100.0)

            self.ann_result.setText(f"{controller_ann_str} with {max_val_ann} %")
            self.cnn_3d_result.setText(f"{controller_3d_cnn_str} with {max_val_3d_cnn} %")
            self.cnn_result.setText("")

            self.average_score.setText(f"{(max_val_ann + max_val_3d_cnn) / 2} %")

            if (max_val_ann + max_val_3d_cnn) / 2 >= 90.0:
                self.final_message.setText("Congratulation! Your form in the exercise is awesome")
            elif (max_val_ann + max_val_3d_cnn) / 2 >= 60 and (max_val_ann + max_val_3d_cnn) / 2 <= 89.9:
                self.final_message.setText("You form could do some work! But you are on the right track!")
            else:
                base_message = " Stop what you are doing. You need to fix your form immediately!\n Here is a video to help you out:\n"
                link_video = ""
                match controller_ann_str:
                    case "ABS":
                        link_video = "https://www.youtube.com/watch?v=k1cLRd7cahQ&ab_channel=MikeThurston"
                    case "BACK":
                        link_video = "https://www.youtube.com/watch?v=2tnATDflg4o&ab_channel=JeremyEthier"
                    case "BICEPS":
                        link_video = "https://www.youtube.com/watch?v=TaeMP1WJTKw&ab_channel=GravityTransformation-FatLossExperts"
                    case "BUTT":
                        link_video = "https://www.youtube.com/watch?v=6vlP9xPJbaQ&ab_channel=KrissyCela"
                    case "CHEST":
                        link_video = "https://www.youtube.com/watch?v=89e518dl4I8&ab_channel=ATHLEAN-X%E2%84%A2"
                    case "FOREARM":
                        link_video = "https://www.youtube.com/watch?v=zVJ3MN4-kkQ&ab_channel=GravityTransformation-FatLossExperts"
                    case "LEGS":
                        link_video = "https://www.youtube.com/watch?v=uYkpTWfpFHA&t=1s&ab_channel=WORKOUT"
                    case "SHOULDER":
                        link_video = "https://www.youtube.com/watch?v=rnDWTXYDMOg&ab_channel=GravityTransformation-FatLossExperts"
                    case "TRICEPS":
                        link_video = "https://www.youtube.com/watch?v=SuajkDYlIRw&ab_channel=VShred"    
                self.final_message.setText(f"{base_message} {link_video} ")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:16pt; font-weight:792;\">Improving Physical Activity Performance with Machine Learning<br />An Automated Image and Video Analysis System</span></p></body></html>"))
        self.insert_button.setText(_translate("MainWindow", "Insert Video / Image"))
        self.textBrowser_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-weight:600;\">How does this work?</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-weight:600;\">Quite simple. Firstly record yourself doing any type of physical exercise.<br />Secondly, press on the &quot;Insert Video&quot; button and select your video.<br /><br />The results will appear at the bottom of the application, with your grades from each Machine Learning model</span></p></body></html>"))
        self.textBrowser_3.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:12pt; font-weight:600;\">ANN RESULT</span></p></body></html>"))
        self.textBrowser_4.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:12pt; font-weight:600;\">CNN RESULT</span></p></body></html>"))
        self.textBrowser_5.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:12pt; font-weight:600;\">3DCNN RESULT</span></p></body></html>"))
        self.ann_result.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'MS Shell Dlg 2\'; font-size:8.25pt;\"><br /></p></body></html>"))
        self.cnn_3d_result.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'MS Shell Dlg 2\'; font-size:8.25pt;\"><br /></p></body></html>"))
        self.cnn_result.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'MS Shell Dlg 2\'; font-size:8.25pt;\"><br /></p></body></html>"))
        self.textBrowser_9.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:12pt; font-weight:600;\">AVERAGE SCORE</span></p></body></html>"))
        self.average_score.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'MS Shell Dlg 2\'; font-size:8.25pt;\"><br /></p></body></html>"))
        self.final_message.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'MS Shell Dlg 2\'; font-size:8.25pt;\"><br /></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
