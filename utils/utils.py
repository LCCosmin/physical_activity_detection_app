from typing import Any, Union
import matplotlib.pyplot as plt
from .constants import ANN_SIZE
import tensorflow as tf
from keras import layers
import keras
import einops
from helpers.enums import TrainerEnum


def transform_file_name_into_int(filename: str) -> list:
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


def transform_int_into_file_name(idx: int) -> str:
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
    if idx == 0:
       return "ABS"
    if idx == 1:
       return "BACK"
    if idx == 2:
       return "BICEPS"
    if idx == 3:
       return "BUTT"
    if idx == 4:
       return "CHEST"
    if idx == 5:
       return "FOREARM"
    if idx == 6:
       return "LEGS"
    if idx == 7:
       return "SHOULDER"
    if idx == 8:
       return "TRICEPS"


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
            if TrainerEnum.ANN.value == signature.value or TrainerEnum.CNN_3D.value == signature.value:
               new_x_data.append(x_training_data[idx])
            else:
              new_x_data.append(transform_npndarray_list_to_list(x_training_data[idx]))
            new_y_data.append(y_training_data[idx])

    return new_x_data, new_y_data


def cut_too_long_training_data(
      x_training_data: list,
      y_training_data:list, 
      limiter: int,
  ) -> Union[list, list]:
    new_x_data = []
    new_y_data = []

    for idx, elem in enumerate(x_training_data):
        if len(elem) >= limiter:
            new_x_data.append(x_training_data[idx][0:limiter])
            new_y_data.append(y_training_data[idx])

    return new_x_data, new_y_data


def create_graph_classes(y_train_data: list, name: str) -> None:
  data = {
      "ABS": 0,
      "BACK": 0,
      "BUTT": 0,
      "CHEST": 0,
      "FOREARM": 0,
      "LEGS": 0,
      "SHOULDER": 0,
      "TRICEPS": 0,
      "BICEPS": 0,
   }

  for elem in y_train_data:
      idx = elem.index(1)
      exercise_name = transform_int_into_file_name(idx)
      data[exercise_name] = data[exercise_name] + 1

  plot_names = list(data.keys())
  plot_values = list(data.values())

  plt.figure(figsize = (10, 5))
  plt.bar(plot_names, plot_values)

  for i in range(len(plot_names)):
      plt.text(i, plot_values[i], plot_values[i])

  plt.xlabel(f"Classes for {name.upper()}")
  plt.ylabel("Number of elements")
  plt.savefig(f"./{name}_classes_diagram.png")
  plt.clf()


def plot_graph(history: Any, name: str) -> None:
    plt.plot(history.history['loss'])
    plt.savefig(f"./{name}_loss.png")
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.savefig(f"./{name}_accuracy.png")
    plt.clf()


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