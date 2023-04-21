from dataclasses import dataclass, field
from utils.constants import WIDTH_PICTURE_CNN, HEIGHT_PICTURE_CNN
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from time import sleep
from utils.utils import plot_graph

@dataclass(kw_only=True)
class PictureModelCNN:
    _width_picture: int = WIDTH_PICTURE_CNN
    _height_picture: int = HEIGHT_PICTURE_CNN
    _epochs_no: int = 1024
    _batch_size: int = 256
    _checkpoint_path: str = field(init=False)
    _training_folder: str = field(init=False)
    __model: tf.keras.models.Sequential = field(init=False)

    
    def __post_init__(self) -> None:
        self._training_folder = "./training_data_picture_model_cnn"
        self._checkpoint_path = "./brains/picture_model_brain.ckpt"
        self.__model = self.create_model()


    def normalize(self, img):
        img = cv2.resize(img, (self._width_picture, self._height_picture), interpolation = cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (1,1), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255

        return img

    
    def create_model(self) -> tf.keras.models.Sequential:
        model = tf.keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(self._width_picture * self._height_picture,)),
            keras.layers.Dense(400, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(200, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(100, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(50, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            
            keras.layers.Dense(9, activation = 'sigmoid')
            ])
        
        model.compile(
            optimizer = 'adam', 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = ['accuracy']
        )

        return model


    def save_model(self) -> None:
        self.__model.save_weights(self._checkpoint_path)


    def train_model(self, x_training_data: list, y_training_data: list) -> None:
        x_training_data, x_test, y_training_data, y_test =(
            train_test_split(
                np.array(x_training_data, dtype=(float)).reshape(-1, self._width_picture * self._height_picture),
                np.array(y_training_data, dtype=(int)), 
                test_size = 0.3,
                shuffle=(True)
            )
        )
        print(f"{len(y_training_data)}")
        sleep(10)
        print(f"{(y_training_data)}")
        sleep(10)
        history = self.__model.fit(x_training_data, y_training_data, epochs = self._epochs_no, batch_size = self._batch_size)

        plot_graph(history, "cnn_model")

        _, accuracy = self.__model.evaluate(x_test, y_test, verbose=2)

        print(f"INFO:SHOWING RESULTS OF THE TRAINING: {self.__model.predict(x_test)[0]}")
        print(f"INFO:MODEL_CNN: Accuracy : {accuracy}")
        print("INFO:MODEL_CNN: Training done ..")


    def evaluate_image(self, image) -> None:
        ...
