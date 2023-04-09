from datetime import datetime
from utils.Controller import ControllerClass
from utils.decorators import benchmark
import cv2
import os

@benchmark
def main():
    controller = ControllerClass(0.3, 0.3, 1)
    
    training_data_ann = controller.gather_data_for_ann()
    print(len(training_data_ann))
    print(len(training_data_ann[0]))
    print(len(training_data_ann[1]))
    print(len(training_data_ann[2]))
    print(len(training_data_ann[3]))
    print(len(training_data_ann[4]))
    print(len(training_data_ann[5]))

    
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
