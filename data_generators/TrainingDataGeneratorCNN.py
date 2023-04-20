import cv2
from typing import Any
import os
import numpy as np


class TrainingDataGeneratorCNN:
    def __init__(
        self,
        vid,
        height_image,
        width_image,
    ) -> None:
        self.__vid = vid
        self.__height_image = height_image
        self.__width_image = width_image


    def normalize(self, img: Any) -> Any:
        img = cv2.resize(img, (self.__width_image, self.__height_image), interpolation = cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (1,1), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = img / 255

        return img

    
    def generate_data(self) -> list:
        final_data = []
        count_frame = 0

        if type(self.__vid) is str:
            self.__vid = cv2.VideoCapture(os.getcwd()  + "/training data/" + self.__vid)
        else:
            self.__vid = cv2.VideoCapture(self.__vid)

        while self.__vid.isOpened():
            success, image = self.__vid.read()

            if not success:
                break

            if count_frame % 960 == 0:
                image = self.normalize(image)
                final_data.append(image)
                count_frame = 0
            else:
                count_frame += 1

        self.__vid.release()
        cv2.destroyAllWindows()
        return final_data


    def update_new_obj(self, filename: str) -> None:
        self.__vid = filename
