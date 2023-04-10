from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


@dataclass(kw_only=True)
class VectorModelANN:
    _vector_size: int = 6
    _epochs_no: int = 256
    _batch_size: int = 32
    _checkpoint_path: str = field(init=False)
    _training_folder: str = field(init=False)

    def __post_init__(self) -> None:
        self._training_folder = "./training_data_vector_model_cnn"
        self._checkpoint_path = "./brains/vector_model_cnn/cp.ckpt"

    def create_model(self):
        model = tf.keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(self._vector_size)),
            keras.layers.Dense(400, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(200, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(100, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(50, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            
            keras.layers.Dense(2, activation = 'sigmoid')
            ])

        model.compile(optimizer = 'adam', 
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics = ['accuracy'])

        return model

    def train_and_save(self, x_training_data, y_training_data) -> None:
        #Get the training data
        print("INFO:VECTOR_MODEL_ANN: Starting the training protocol ...")

        x_data = np.array(x_training_data)
        y_data = np.array(y_training_data)

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, 
                                                            test_size = 0.2, shuffle=(True))

        model = self.create_model()

        history = model.fit(x_train, y_train, epochs = self._epochs_no, batch_size = self._batch_size)
        model.save_weights(self._checkpoint_path)

        loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

        print(model.predict(x_test)[0])

        plt.plot(history.history['loss'])
        plt.show(block=True)
        plt.savefig("./ann_loss.png")

        print("INFO:VECTOR_MODEL_ANN: Accuracy : " + str(accuracy))

        print("INFO:VECTOR_MODEL_ANN: Training done ..")

    def evaluate_video(self, video) -> None:
        model = self.create_model()
        model.load_weights(self._checkpoint_path).expect_partial()

        # n = 1

        # for person in people_list:
        #     copy_image = person
        #     person = self.normalize(person)
        #     person = np.array(cv2.resize(person, (self._width_crop, self._height_crop)), dtype = float).reshape(-1, self._width_crop * self._height_crop)
            
        #     predictions = model.predict(person)
        #     if predictions[0][0] > 0.5:
        #         cv2.imwrite("./saves/persons/non-military/image-video{}.jpg".format(n), copy_image)
        #     else:
        #         cv2.imwrite("./saves/persons/military/image-video{}.jpg".format(n), copy_image)
        #     n = n + 1
