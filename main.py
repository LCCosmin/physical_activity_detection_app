from controller.Controller import ControllerClass
from utils.utils import transform_initial_x_data, cut_too_short_training_data
import cv2
import os
from utils.constants import WIDTH_3D_CNN, HEIGHT_3D_CNN, ANN_SIZE, MIN_NUMBER_OF_FRAMES_IN_3D_CNN


def main():
    controller = ControllerClass(
        detection_confidence_gen_ann=0.3,
        tracking_confidence_gen_ann=0.3,
        complexity_gen_ann=1,
        width_3d_cnn = WIDTH_3D_CNN,
        height_3d_cnn = HEIGHT_3D_CNN,
    )
    
    # x_train_data_ann, y_train_data_ann = controller.gather_data_for_ann()
    # x_train_data_ann = transform_initial_x_data(x_train_data_ann)
    # x_train_data_ann, y_train_data_ann = cut_too_short_training_data(
    #     x_training_data=x_train_data_ann, 
    #     y_training_data=y_train_data_ann, 
    #     limiter=ANN_SIZE*6
    # )

    x_train_data_3d_cnn, y_train_data_3d_cnn = controller.gather_data_for_3d_cnn()
    x_train_data_3d_cnn, y_train_data_3d_cnn = cut_too_short_training_data(
        x_training_data=x_train_data_3d_cnn,
        y_training_data=y_train_data_3d_cnn, 
        limiter=MIN_NUMBER_OF_FRAMES_IN_3D_CNN
    )

    # controller.train_ann(x_train_data_ann, y_train_data_ann)
    
    
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
