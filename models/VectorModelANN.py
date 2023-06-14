from dataclasses import dataclass, field
from typing import Any, Tuple
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils.constants import ANN_SIZE
from utils.utils import plot_graph, transform_initial_x_data, transform_int_into_file_name
from data_generators.TrainingDataGeneratorANN import TrainingDataGeneratorANN
from helpers.enums import TrainerAction


@dataclass(kw_only=True)
class VectorModelANN:
    _vector_size: int = ANN_SIZE * 6
    _epochs_no: int = 16
    _batch_size: int = 16
    _checkpoint_path: str = field(init=False)
    _training_folder: str = field(init=False)
    __model: tf.keras.models.Sequential = field(init=False)


    def __post_init__(self) -> None:
        self._checkpoint_path = "./brains/ann_brain/vector_model_brain.ckpt"
        self.__model = self.create_model()


    def create_model(self) -> tf.keras.models.Sequential:
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
            
            keras.layers.Dense(9, activation = 'softmax')
            ])

        model.compile(
            optimizer = 'adam', 
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics = ['accuracy'],
        )

        return model


    def save_model(self) -> None:
        self.__model.save_weights(self._checkpoint_path)


    def train_model(self, x_training_data: list, y_training_data: list) -> None:
        print("INFO:VECTOR_MODEL_ANN: Starting the training protocol ...")

        x_data = np.array(x_training_data)
        y_data = np.array(y_training_data)

        x_train, x_test, y_train, y_test =(
            train_test_split(
                x_data, y_data, 
                test_size = 0.2,
                shuffle=(True)
            )
        )

        # print(x_data)
        print(y_data)

        history = self.__model.fit(x_train, y_train, epochs = self._epochs_no, batch_size = self._batch_size)

        _, accuracy = self.__model.evaluate(x_test, y_test, verbose=2)

        print(f"INFO:SHOWING RESULTS OF THE TRAINING: {self.__model.predict(x_test)[0]}")
        print(f"INFO:VECTOR_MODEL_ANN: Accuracy : {accuracy}")
        print("INFO:VECTOR_MODEL_ANN: Training done ..")
        plot_graph(history, "ann_plot")


    # @TODO: Must refactor this later
    def evaluate_video(self, vid) -> Tuple[str, list]:
        model = self.create_model()
        model.load_weights(self._checkpoint_path).expect_partial()

        data_image = TrainingDataGeneratorANN(vid=vid)
        data_image = data_image.generate_data(TrainerAction.EVALUATE)
        v = []
        v.append(data_image)

        data_image = transform_initial_x_data(v)
        data_image = np.array(data_image)

        pred = model.predict(data_image)[0]
        max_val = max(pred)
        max_idx = pred.tolist().index(max_val)
        return transform_int_into_file_name(max_idx), pred
