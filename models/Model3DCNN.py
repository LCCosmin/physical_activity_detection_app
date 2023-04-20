from utils.constants import WIDTH_3D_CNN, HEIGHT_3D_CNN
from dataclasses import dataclass, field
from utils.utils import Conv2Plus1D, ResizeVideo, add_residual_block
import keras
from keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
    
@dataclass(kw_only=True)
class Model3DCNN:
  _width_3d_cnn: int = WIDTH_3D_CNN
  _height_3d_cnn: int = HEIGHT_3D_CNN
  _epochs_no: int = 256
  _batch_size: int = 32
  _checkpoint_path: str = field(init=False)
  _training_folder: str = field(init=False)
  __model: keras.Model = field(init=False)


  def __post_init__(self) -> None:
    self._training_folder = "./training_data_vector_model_cnn"
    self._checkpoint_path = "./brains/vector_model_cnn/cp.ckpt"
    self.__model = self.create_model()


  def create_model(self) -> keras.Model:
    input_shape = (None, 10, self._height_3d_cnn, self._width_3d_cnn, 3)
    input = layers.Input(shape=(input_shape[1:]))
    x = input

    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ResizeVideo(self._height_3d_cnn // 2, self._width_3d_cnn // 2)(x)

    # Block 1
    x = add_residual_block(x, 16, (3, 3, 3))
    x = ResizeVideo(self._height_3d_cnn // 4, self._width_3d_cnn // 4)(x)

    # Block 2
    x = add_residual_block(x, 32, (3, 3, 3))
    x = ResizeVideo(self._height_3d_cnn // 8, self._width_3d_cnn // 8)(x)

    # Block 3
    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeVideo(self._height_3d_cnn // 16, self._width_3d_cnn // 16)(x)

    # Block 4
    x = add_residual_block(x, 128, (3, 3, 3))

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10)(x)

    model = keras.Model(input, x)
    
    model.compile(
      loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
      optimizer = keras.optimizers.Adam(learning_rate = 0.0001), 
      metrics = ['accuracy']
    )

    return model


  def save_model(self) -> None:
    self.__model.save_weights(self._checkpoint_path)
    
    
  def evaluate_video(self, video) -> None:
    self.__model.load_weights(self._checkpoint_path).expect_partial()
    
  
  def train_model(self, x_training_data: list, y_training_data: list) -> None:
    print("INFO:3D_MODEL_CNN: Starting the training protocol ...")

    # print(f"""
    #   \n\n
    #     {type(x_training_data[0][0])}\n
    #     ------------------
    #     \n
    #     {type(x_training_data)}
    #     \n\n
    #     {y_training_data}
    #   \n\n   
    # """)

    # x_data = np.array(x_training_data)
    # y_data = np.array(y_training_data)
    x_data = x_training_data
    y_data = y_training_data

    x_train, x_test, y_train, y_test =(
        train_test_split(
            x_data, y_data, 
            test_size = 0.2,
            shuffle=(True)
        )
    )
    
    
    
    history = self.__model.fit(x_train, y_train, epochs = self._epochs_no, batch_size = self._batch_size)

    _, accuracy = self.__model.evaluate(x_test, y_test, verbose=2)

    print(f"INFO:SHOWING RESULTS OF THE TRAINING: {self.__model.predict(x_test)[0]}")
    print(f"INFO:3D_MODEL_CNN: Accuracy : {accuracy}")
    print("INFO:3D_MODEL_CNN: Training done ..")
