from utils.constants import WIDTH_3D_CNN, HEIGHT_3D_CNN
from dataclasses import dataclass, field
import tensorflow as tf


@dataclass(kw_only=True)
class Model3DCNN:
    _width_3d_cnn: int = WIDTH_3D_CNN
    _height_3d_cnn: int = HEIGHT_3D_CNN
    _epochs_no: int = 256
    _batch_size: int = 32
    _checkpoint_path: str = field(init=False)
    _training_folder: str = field(init=False)
    __model: tf.keras.models.Sequential = field(init=False)

    # def create_model(self) -> None:

    #     input_shape = (None, 10, HEIGHT, WIDTH, 3)
    #     input = layers.Input(shape=(input_shape[1:]))
    #     x = input

    #     x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
    #     x = layers.BatchNormalization()(x)
    #     x = layers.ReLU()(x)
    #     x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

    #     # Block 1
    #     x = add_residual_block(x, 16, (3, 3, 3))
    #     x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

    #     # Block 2
    #     x = add_residual_block(x, 32, (3, 3, 3))
    #     x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

    #     # Block 3
    #     x = add_residual_block(x, 64, (3, 3, 3))
    #     x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

    #     # Block 4
    #     x = add_residual_block(x, 128, (3, 3, 3))

    #     x = layers.GlobalAveragePooling3D()(x)
    #     x = layers.Flatten()(x)
    #     x = layers.Dense(10)(x)

    #     model = keras.Model(input, x)