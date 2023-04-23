from typing import Any, Union
import matplotlib.pyplot as plt
from .constants import ANN_SIZE
import tensorflow as tf
from keras import layers
import keras
import einops
from helpers.enums import TrainerEnum


def transfor_file_name_into_int(filename: str) -> list:
        """_summary_

            0 - ABS
            1 - BACK
            2 - BICEPS
            3 - BUTT
            4 - CHEST
            5 - FOREARM
            6 - LEGS
            7 - SHOULDER
            8 - TRICEPS
        """
        if "abs" in filename:
            return [1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif "back" in filename:
            return [0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif "biceps" in filename:
            return [0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif "butt" in filename:
            return [0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif "chest" in filename:
            return [0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif "forearm" in filename:
            return [0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif "legs" in filename:
            return [0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif "shoulder" in filename:
            return [0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif "triceps" in filename:
            return [0, 0, 0, 0, 0, 0, 0, 0, 1]


def transform_initial_x_data(x_training_data: list) -> list:
    final_x_data = []

    for exercise in x_training_data:
        vector_slice_0 = []
        vector_slice_1 = []
        vector_slice_2 = []
        vector_slice_3 = []
        vector_slice_4 = []
        vector_slice_5 = []
        exercise = exercise[0:ANN_SIZE] if len(exercise) > ANN_SIZE else exercise

        for slice in exercise:
            vector_slice_0.append(slice[0])
            vector_slice_1.append(slice[1])
            vector_slice_2.append(slice[2])
            vector_slice_3.append(slice[3])
            vector_slice_4.append(slice[4])
            vector_slice_5.append(slice[5])
        
        final_x_data.append(
            vector_slice_0 
            + vector_slice_1
            + vector_slice_2
            + vector_slice_3
            + vector_slice_4
            + vector_slice_5
        )

    return final_x_data


def transform_npndarray_list_to_list(array: list) -> list:
    return [elem.tolist() for elem in array]


def cut_too_short_training_data(
      x_training_data: list,
      y_training_data:list, 
      limiter: int,
      signature: TrainerEnum
  ) -> Union[list, list]:
    new_x_data = []
    new_y_data = []

    for idx, elem in enumerate(x_training_data):
        if len(elem) >= limiter:
            if TrainerEnum.ANN.value == signature.value:
               new_x_data.append(x_training_data[idx])
            else:
              new_x_data.append(transform_npndarray_list_to_list(x_training_data[idx]))
            new_y_data.append(y_training_data[idx])

    return new_x_data, new_y_data


def plot_graph(history: Any, name: str) -> None:
    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.savefig(f"./{name}.png")


# 3D Model utils
class Conv2Plus1D(keras.layers.Layer):
  def __init__(self, filters, kernel_size, padding):
    """
      A sequence of convolutional layers that first apply the convolution operation over the
      spatial dimensions, and then the temporal dimension. 
    """
    super().__init__()
    self.seq = keras.Sequential([  
        # Spatial decomposition
        layers.Conv3D(filters=filters,
                      kernel_size=(1, kernel_size[1], kernel_size[2]),
                      padding=padding),
        # Temporal decomposition
        layers.Conv3D(filters=filters, 
                      kernel_size=(kernel_size[0], 1, 1),
                      padding=padding)
        ])

  def call(self, x):
    return self.seq(x)


class ResidualMain(keras.layers.Layer):
  """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
  """
  def __init__(self, filters, kernel_size):
    super().__init__()
    self.seq = keras.Sequential([
        Conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization(),
        layers.ReLU(),
        Conv2Plus1D(filters=filters, 
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)


class Project(keras.layers.Layer):
  """
    Project certain dimensions of the tensor as the data is passed through different 
    sized filters and downsampled. 
  """
  def __init__(self, units):
    super().__init__()
    self.seq = keras.Sequential([
        layers.Dense(units),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)


def add_residual_block(input, filters, kernel_size):
  """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
  out = ResidualMain(filters, 
                     kernel_size)(input)

  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])


class ResizeVideo(keras.layers.Layer):
  def __init__(self, height, width):
    super().__init__()
    self.height = height
    self.width = width
    self.resizing_layer = layers.Resizing(self.height, self.width)

  def call(self, video):
    """
      Use the einops library to resize the tensor.  

      Args:
        video: Tensor representation of the video, in the form of a set of frames.

      Return:
        A downsampled size of the video according to the new height and width it should be resized to.
    """
    # b stands for batch size, t stands for time, h stands for height, 
    # w stands for width, and c stands for the number of channels.
    old_shape = einops.parse_shape(video, 'b t h w c')
    images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
    images = self.resizing_layer(images)
    videos = einops.rearrange(
        images, '(b t) h w c -> b t h w c',
        t = old_shape['t'])
    return videos