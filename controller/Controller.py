from dataclasses import dataclass
from typing import Union
from data_generators.TrainingDataGeneratorANN import TrainingDataGeneratorANN
from data_generators.TrainingDataGenerator3DCNN import TrainingDataGenerator3DCNN
from utils.decorators import benchmark
from models.VectorModelANN import VectorModelANN
import os
from utils.utils import transfor_file_name_into_int
from helpers.helpers import TrainerANNData, Trainer3DCNNData
from helpers.enums import TrainerEnum


@dataclass
class ControllerClass:
    __ann_model: VectorModelANN
    __ann_data: TrainerANNData
    __3d_cnn_data: Trainer3DCNNData


    def __init__(
        self, 
        ann_data,
        cnn_3d_data,
    ) -> None:
        self.__ann_data = ann_data
        self.__3d_cnn_data = cnn_3d_data

        self.__ann_model = VectorModelANN()


    @benchmark
    def gather_training_data(self, trainer_enum: TrainerEnum) -> Union[list, list]:
        x_training_data = []
        y_training_data = []
        counter_files = 1

        if trainer_enum.value == TrainerEnum.CNN_3D.value:
            trainer = TrainingDataGenerator3DCNN(
                vid='',
                cnn_3d_width=self.__3d_cnn_data.width_3d_cnn,
                cnn_3d_height=self.__3d_cnn_data.height_3d_cnn,
            )
        elif trainer_enum.value == TrainerEnum.ANN.value:
            trainer = TrainingDataGeneratorANN(
                vid='',
                detection_confidence=self.__ann_data.detection_confidence,
                tracking_confidence=self.__ann_data.tracking_confidence,
                complexity=self.__ann_data.complexity,
            )
        else:
            trainer_enum = ...

        for filename in os.listdir(os.getcwd()  + "/training data"):         
            print(f"INFO:GATHER_DATA_{trainer_enum.value}: Processing file {counter_files} with the name {filename}")
            counter_files+=1

            trainer.update_new_vid(filename)

            x_train_slice = trainer.generate_data()

            x_training_data.append(
                x_train_slice
            )
            y_training_data.append(
                transfor_file_name_into_int(filename)
            ) 
        
        return x_training_data, y_training_data


    @benchmark
    def train_ann(self, x_training_data: list, y_training_data: list) -> None:
        self.__ann_model.train_model(x_training_data, y_training_data)


    @benchmark
    def save_ann(self) -> None:
        self.__ann_model.save_model()
