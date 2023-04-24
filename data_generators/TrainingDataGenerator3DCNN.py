from typing import Any
import cv2
import os
import numpy as np
from time import sleep

class TrainingDataGenerator3DCNN:
    def __init__(
        self, 
        vid,
        cnn_3d_width,
        cnn_3d_height,
    ) -> None:
        self.__vid = vid
        self.__3d_cnn_width = cnn_3d_width
        self.__3d_cnn_height = cnn_3d_height


    def normalize(self, img: Any) -> Any:
        img = cv2.resize(img, (self.__3d_cnn_width, self.__3d_cnn_height), interpolation = cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (1,1), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = img / 255
        # img = img.tolist()

        return img


    def generate_data(self) -> list:
        final_3d_data = None

        if type(self.__vid) is str:
            self.__vid = cv2.VideoCapture(os.getcwd()  + "/training data/" + self.__vid)
        else:
            self.__vid = cv2.VideoCapture(self.__vid)

        count_frame = 0
        while self.__vid.isOpened():
            success, image = self.__vid.read()

            if not success:
                break

            if count_frame % 960 == 0:
                image = self.normalize(image)
                if final_3d_data is None:
                    final_3d_data = np.array([image])
                    
                else:            
                    final_3d_data = np.append(final_3d_data, [image], axis=0)
                count_frame = 0
            else:
                count_frame += 1

        self.__vid.release()
        cv2.destroyAllWindows()
        return final_3d_data


    def update_new_obj(self, filename: str) -> None:
        self.__vid = filename
