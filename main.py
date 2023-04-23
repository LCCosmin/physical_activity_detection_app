from controller.Controller import ControllerClass
from utils.utils import transform_initial_x_data, cut_too_short_training_data
import cv2
import os
from utils.constants import (
    WIDTH_3D_CNN,
    HEIGHT_3D_CNN, 
    ANN_SIZE, 
    MIN_NUMBER_OF_FRAMES_IN_3D_CNN, 
    WIDTH_PICTURE_CNN, 
    HEIGHT_PICTURE_CNN,
    DETECTION_CONFIDENCE_ANN,
    TRACKING_CONFIDENCE_ANN,
    COMPLEXITY_ANN,
)
from helpers.helpers import TrainerANNData, Trainer3DCNNData, TrainerCNNData
from helpers.enums import TrainerEnum
import numpy as np


def main():
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
    # controller.train_ann(x_train_data_ann, y_train_data_ann)
    # controller.save_ann()
    # print(f"Predict result for ANN is: {controller.evaluate_ann_image('/home/cosmin/Desktop/licenta/physical_activity_detection_app/111.png')}")

    # 3D CNN
    # x_train_data_3d_cnn, y_train_data_3d_cnn = controller.gather_training_data(TrainerEnum.CNN_3D)
    # x_train_data_3d_cnn, y_train_data_3d_cnn = cut_too_short_training_data(
    #     x_training_data=x_train_data_3d_cnn,
    #     y_training_data=y_train_data_3d_cnn, 
    #     limiter=MIN_NUMBER_OF_FRAMES_IN_3D_CNN
    #     signature=TrainerEnum.CNN_3D
    # )
    # controller.train_3d_cnn(x_train_data_3d_cnn, y_train_data_3d_cnn)

    # CNN
    x_train_data_cnn, y_train_data_cnn = controller.gather_training_data(TrainerEnum.CNN)
    controller.train_cnn(x_train_data_cnn, y_train_data_cnn)
    controller.save_cnn()
    controller.evaluate_cnn_image("/home/cosmin/Desktop/licenta/physical_activity_detection_app/111.png")
    
from time import sleep

def test_video():
    vid = f"{os.getcwd()}/training data/abs_1.mp4"
    
    vid = cv2.VideoCapture(vid)
    while vid.isOpened():
        suc, img = vid.read()
        
        if suc == False:
            break

        img = img.tolist()
        print(f"suc: {type(suc)}")
        print(f"img: {type(img)}")

        cv2.imshow("d", img)
        cv2.waitKey(0)
        
    print(os.getcwd())
    vid.release()
    cv2.destroyAllWindows()

def reverse_videos():
    total_files = len([f for f in os.listdir("C:\\Users\\Cosmin\\Desktop\\licenta\\training data") if f.endswith("mp4")])
    # Iterate over files of given type in input directory
    for c, filename in enumerate([f for f in os.listdir("C:\\Users\\Cosmin\\Desktop\\licenta\\training data") if
                                f.endswith("mp4")]):
        print("Processing file '%s' (%s of %s)." % (filename, c+1,
            total_files))
        video = cv2.VideoCapture("C:\\Users\\Cosmin\\Desktop\\licenta\\training data\\" + filename)
        # Gather info about input video
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object for output
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fn, ext = os.path.splitext(os.path.basename(filename))
        out = cv2.VideoWriter("%s/%s_%s%s" % ("C:\\Users\\Cosmin\\Desktop\\licenta\\rev", fn, "rev_", ext),
                            fourcc, fps, (width, height))

        # Flip video frame by frame and write to output file
        while(video.isOpened()):
            ret, frame = video.read()
            if ret:
                frame = cv2.flip(frame, 1)
                out.write(frame)
            else:
                break

    video.release()
    out.release()
    
def rename_files():
    START = "chest_"
    # Iterate over files of given type in input directory
    for c, filename in enumerate([f for f in os.listdir("C:\\Users\\Cosmin\\Desktop\\licenta\\training data") if
                                f.endswith("mp4")]):
        new_file = START + str(c+1) + ".mp4"
        os.rename("C:\\Users\\Cosmin\\Desktop\\licenta\\training data\\" + filename, "C:\\Users\\Cosmin\\Desktop\\licenta\\renamed-files\\" + new_file)
    
if __name__ == "__main__":
    #rename_files()
    #reverse_videos()
    main()
    # test_video()
