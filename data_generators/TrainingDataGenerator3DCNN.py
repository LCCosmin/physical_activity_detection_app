from typing import Any
import cv2

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

        return img


    def generate_3d_data(self) -> list:
        final_3d_data = []

        if type(self.__vid) is str:
            self.__vid = cv2.VideoCapture("./training data/" + self.__vid)
        else:
            self.__vid = cv2.VideoCapture(self.__vid)

        while self.__vid.isOpened():
            success, image = self.__vid.read()

            if not success:
                break

            image = self.normalize(image)
            final_3d_data.append(image)
        
        return final_3d_data
