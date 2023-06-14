from controller.Controller import ControllerClass
from utils.constants import (
    WIDTH_3D_CNN,
    HEIGHT_3D_CNN, 
    WIDTH_PICTURE_CNN, 
    HEIGHT_PICTURE_CNN,
    DETECTION_CONFIDENCE_ANN,
    TRACKING_CONFIDENCE_ANN,
    COMPLEXITY_ANN,
)
from helpers.helpers import TrainerANNData, Trainer3DCNNData, TrainerCNNData
from PyQt5 import QtCore, QtGui, QtWidgets
from UI.main_window import Ui_MainWindow
import sys


def main_test():
    trainer_ann_data = TrainerANNData(
        detection_confidece=DETECTION_CONFIDENCE_ANN,
        tracking_confidence=TRACKING_CONFIDENCE_ANN,
        complexity=COMPLEXITY_ANN,
    )

    trainer_3d_cnn_data = Trainer3DCNNData(
        width_3d_cnn = WIDTH_3D_CNN,
        height_3d_cnn = HEIGHT_3D_CNN,
    )

    trainer_cnn_data = TrainerCNNData(
        width_cnn=WIDTH_PICTURE_CNN,
        height_cnn=HEIGHT_PICTURE_CNN,
    )

    controller = ControllerClass(
        ann_data=trainer_ann_data,
        cnn_3d_data=trainer_3d_cnn_data,
        cnn_data=trainer_cnn_data,
    )
    
    # ANN
    # x_train_data_ann, y_train_data_ann = controller.gather_training_data(TrainerEnum.ANN)
    # x_train_data_ann = transform_initial_x_data(x_train_data_ann)
    # x_train_data_ann, y_train_data_ann = cut_too_short_training_data(
    #     x_training_data=x_train_data_ann, 
    #     y_training_data=y_train_data_ann, 
    #     limiter=ANN_SIZE*6,
    #     signature=TrainerEnum.ANN
    # )
    # create_graph_classes(y_train_data_ann, "ann")
    # controller.train_ann(x_train_data_ann, y_train_data_ann)
    # controller.save_ann()
    print("\n\nANN")
    print("\n\nUNSEEN EXERCISES PREDICTION:\n\n")
    print(f"Predict result (LEGS) for ANN is: {controller.evaluate_ann_video('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/squats.mp4')}")
    print(f"Predict result (CHEST) for ANN is: {controller.evaluate_ann_video('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/pushup.mp4')}")
    print(f"Predict result (ABS) for ANN is: {controller.evaluate_ann_video('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/plank.mp4')}")
    print(f"Predict result (ABS) for ANN is: {controller.evaluate_ann_video('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/abs.mp4')}")
    print("\n\nSEEN EXERCISES PREDICTION:\n\n")
    print(f"Predict result (ABS) for ANN is: {controller.evaluate_ann_video('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/seen_abs.mp4')}")
    print(f"Predict result (BACK) for ANN is: {controller.evaluate_ann_video('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/seen_back.mp4')}")
    print(f"Predict result (SHOULDER) for ANN is: {controller.evaluate_ann_video('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/seen_shoulder.mp4')}")
    print(f"Predict result (FOREARM) for ANN is: {controller.evaluate_ann_video('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/seen_forearm.mp4')}")

    # 3D CNN
    # x_train_data_3d_cnn, y_train_data_3d_cnn = controller.gather_training_data(TrainerEnum.CNN_3D)
    # x_train_data_3d_cnn, y_train_data_3d_cnn = cut_too_short_training_data(
    #     x_training_data=x_train_data_3d_cnn,
    #     y_training_data=y_train_data_3d_cnn, 
    #     limiter=MIN_NUMBER_OF_FRAMES_IN_3D_CNN,
    #     signature=TrainerEnum.CNN_3D
    # )
    # x_train_data_3d_cnn, y_train_data_3d_cnn = cut_too_long_training_data(
    #     x_training_data=x_train_data_3d_cnn,
    #     y_training_data=y_train_data_3d_cnn, 
    #     limiter=MIN_NUMBER_OF_FRAMES_IN_3D_CNN,
    # )
    # create_graph_classes(y_train_data_3d_cnn, "cnn_3d")
    # controller.train_3d_cnn(x_train_data_3d_cnn, y_train_data_3d_cnn)
    # controller.save_3d_cnn()
    print("\n\n3D CNN")
    print("\n\nUNSEEN EXERCISES PREDICTION:\n\n")
    print(f"Predict result (LEGS) for 3D CNN is: {controller.evaluate_3d_cnn_video('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/squats.mp4')}")
    print(f"Predict result (CHEST) for 3D CNN is: {controller.evaluate_3d_cnn_video('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/pushup.mp4')}")
    print(f"Predict result (ABS) for 3D CNN is: {controller.evaluate_3d_cnn_video('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/plank.mp4')}")
    print(f"Predict result (ABS) for 3D CNN is: {controller.evaluate_3d_cnn_video('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/abs.mp4')}")
    print("\n\SEEN EXERCISES PREDICTION:\n\n")
    print(f"Predict result (ABS) for 3D CNN is: {controller.evaluate_3d_cnn_video('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/seen_abs.mp4')}")
    print(f"Predict result (BACK) for 3D CNN is: {controller.evaluate_3d_cnn_video('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/seen_back.mp4')}")
    print(f"Predict result (SHOULDER) for 3D CNN is: {controller.evaluate_3d_cnn_video('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/seen_shoulder.mp4')}")
    print(f"Predict result (FOREARM) for 3D CNN is: {controller.evaluate_3d_cnn_video('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/seen_forearm.mp4')}")


    # CNN
    # create_graph_classes([[0,1]], "cnn")
    # x_train_data_cnn, y_train_data_cnn = controller.gather_training_data(TrainerEnum.CNN)
    # create_graph_classes(y_train_data_cnn, "cnn")
    # print(f"INFO: LENGTH OF DATASET: {len(y_train_data_cnn)}")
    # controller.train_cnn(x_train_data_cnn, y_train_data_cnn)
    # controller.save_cnn()
    print("\n\nCNN")
    print("\n\nUNSEEN EXERCISES PREDICTION:\n\n")
    print(f"Predict result (LEGS) for CNN is: {controller.evaluate_cnn_image('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/squats.png')}")
    print(f"Predict result (CHEST) for CNN is: {controller.evaluate_cnn_image('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/pushup.png')}")
    print(f"Predict result (ABS) for CNN is: {controller.evaluate_cnn_image('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/plank.png')}")
    print(f"Predict result (ABS) for CNN is: {controller.evaluate_cnn_image('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/abs.png')}")
    print("\n\nSEEN EXERCISES PREDICTION:\n\n")
    print(f"Predict result (ABS) for 3D CNN is: {controller.evaluate_cnn_image('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/seen_abs.png')}")
    print(f"Predict result (BACK) for 3D CNN is: {controller.evaluate_cnn_image('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/seen_back.png')}")
    print(f"Predict result (SHOULDER) for 3D CNN is: {controller.evaluate_cnn_image('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/seen_shoulder.png')}")
    print(f"Predict result (FOREARM) for 3D CNN is: {controller.evaluate_cnn_image('/home/cosmin/Desktop/licenta/x/physical_activity_detection_app/predict_data/seen_forearm.png')}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
