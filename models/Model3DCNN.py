from typing import Optional, Tuple
from utils.constants import WIDTH_3D_CNN, HEIGHT_3D_CNN, MIN_NUMBER_OF_FRAMES_IN_3D_CNN
from dataclasses import dataclass, field
from utils.utils import Conv2Plus1D, ResizeVideo, add_residual_block, plot_graph
import keras
from keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from time import sleep
import cv2
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from keras.models import Sequential



@dataclass(kw_only=True)
class Model3DCNN:
    _width_picture: int = WIDTH_3D_CNN
    _height_picture: int = HEIGHT_3D_CNN
    _batch_size: int = 32
    _epochs: int = 256
    _checkpoint_path: str = field(init=False)
    __model: Optional[tf.keras.models.Sequential] = field(init=False)


    def __post_init__(self) -> None:
        self._checkpoint_path = "./brains/cnn_3d_brain/cnn_3d_model_brain.ckpt"
        self.__model = self.create_model()
    
  
    def normalize(self, img):
        img = cv2.resize(img, (self._width_picture, self._height_picture), interpolation = cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (1,1), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255

        return img
  

    def create_model(self) -> tf.keras.models.Sequential:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(MIN_NUMBER_OF_FRAMES_IN_3D_CNN, self._height_picture, self._width_picture, 1)),
            tf.keras.layers.MaxPooling3D((2, 2, 2)),
            tf.keras.layers.BatchNormalization(center=True, scale=True),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
            tf.keras.layers.MaxPooling3D((2, 2, 2)),
            tf.keras.layers.BatchNormalization(center=True, scale=True),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu'),
            tf.keras.layers.MaxPooling3D((2, 2, 2)),
            tf.keras.layers.BatchNormalization(center=True, scale=True),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu'),
            tf.keras.layers.MaxPooling3D((2, 2, 2)),
            tf.keras.layers.BatchNormalization(center=True, scale=True),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(9, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.Accuracy()]
        )
        
        return model
    

    def save_model(self) -> None:
        self.__model.save_weights(self._checkpoint_path)


    def load_model(self) -> None:
        self.__model.load_weights(self._checkpoint_path)


    def train_model(self, x_train_data: list, y_train_data: list) -> None:
        print("INFO:TRAIN_3D_CNN: TRAINING IS STARTING...")
        x_train_data = [x.reshape(MIN_NUMBER_OF_FRAMES_IN_3D_CNN, self._height_picture, self._width_picture, 1) for x in x_train_data]
        x_train_data = np.expand_dims(x_train_data, axis=-1)
        y_train_data = np.array(y_train_data, dtype=int)

        x_train_data, x_test, y_train_data, y_test = train_test_split(
            x_train_data,
            y_train_data,
            test_size=0.3,
            shuffle=True
        )
        
        # Used CPU training
        history = self.__model.fit(x_train_data, y_train_data, epochs=self._epochs, batch_size=self._batch_size, use_multiprocessing=True)

        plot_graph(history, "cnn_3d_model")

        _, accuracy = self.__model.evaluate(x_test, y_test, verbose=2)

        print(f"INFO:SHOWING RESULTS OF THE TRAINING: {self.__model.predict(x_test)[0]}")
        print(f"INFO:MODEL_CNN: Accuracy : {accuracy}")
        print("INFO:MODEL_CNN: Training done ..")


    def evaluate(self, test_dataset: tf.data.Dataset) -> Tuple[float, float]:
        loss, acc = self.__model.evaluate(test_dataset)
        return loss, acc


    def predict(self, x: tf.Tensor) -> tf.Tensor:
        return self.__model.predict(x)
