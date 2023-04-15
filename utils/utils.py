from typing import Any, Union
import matplotlib.pyplot as plt
from .constants import ANN_SIZE

def transfor_file_name_into_int(filename: str) -> int:
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
            return 0
        elif "back" in filename:
            return 1
        elif "biceps" in filename:
            return 2
        elif "butt" in filename:
            return 3
        elif "chest" in filename:
            return 4
        elif "forearm" in filename:
            return 5
        elif "legs" in filename:
            return 6
        elif "shoulder" in filename:
            return 7
        elif "triceps" in filename:
            return 8


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


def cut_too_short_training_data(x_training_data: list, y_training_data:list, limiter: int) -> Union[list, list]:
    new_x_data = []
    new_y_data = []

    for idx, elem in enumerate(x_training_data):
        if len(elem) >= limiter:
            new_x_data.append(x_training_data[idx])
            new_y_data.append(y_training_data[idx])

    return new_x_data, new_y_data


def plot_graph(history: Any) -> None:
    plt.plot(history.history['loss'])
    plt.savefig("./ann_loss.png")
