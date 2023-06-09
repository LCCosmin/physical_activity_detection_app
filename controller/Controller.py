from dataclasses import dataclass
from typing import Union, Tuple
from data_generators.TrainingDataGeneratorANN import TrainingDataGeneratorANN
from data_generators.TrainingDataGenerator3DCNN import TrainingDataGenerator3DCNN
from data_generators.TrainingDataGeneratorCNN import TrainingDataGeneratorCNN
from utils.decorators import benchmark
from models.VectorModelANN import VectorModelANN
from models.Model3DCNN import Model3DCNN
from models.PictureModelCNN import PictureModelCNN
import os
from utils.utils import transform_file_name_into_int, transform_int_into_file_name
from helpers.helpers import TrainerANNData, Trainer3DCNNData, TrainerCNNData
from helpers.enums import TrainerEnum, TrainerAction


@dataclass
class ControllerClass:
    # models
    __ann_model: VectorModelANN
    __3d_cnn_model: Model3DCNN
    __cnn_model: PictureModelCNN
    # data for models
    __ann_data: TrainerANNData
    __3d_cnn_data: Trainer3DCNNData
    __cnn_data: TrainerCNNData


    def __init__(
        self, 
        ann_data,
        cnn_3d_data,
        cnn_data,
    ) -> None:
        self.__ann_data = ann_data
        self.__3d_cnn_data = cnn_3d_data
        self.__cnn_data = cnn_data

        self.__ann_model = VectorModelANN()
        self.__3d_cnn_model = Model3DCNN()
        self.__cnn_model = PictureModelCNN()


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
            trainer = TrainingDataGeneratorCNN(
                vid='',
                width_image=self.__cnn_data.width_cnn,
                height_image=self.__cnn_data.height_cnn,
            )

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

        for filename in os.listdir(os.getcwd()  + "/training data"):         
            print(f"INFO:GATHER_DATA_{trainer_enum.value}: Processing file {counter_files} with the name {filename}")
            idx = transform_file_name_into_int(filename).index(1)
            exercise_name = transform_int_into_file_name(idx)
            data[exercise_name] = data[exercise_name] + 1
            counter_files+=1

            if data[exercise_name] <=55:
                trainer.update_new_obj(filename)

                x_train_slice = trainer.generate_data(type_mode=TrainerAction.TRAIN)

                if trainer_enum.value == TrainerEnum.CNN.value:
                    x_training_data.extend(
                        x_train_slice
                    )
                    for _ in range(len(x_train_slice)):
                        y_training_data.append(
                            transform_file_name_into_int(filename)
                        )
                else:
                    x_training_data.append(
                        x_train_slice
                    )
                    y_training_data.append(
                        transform_file_name_into_int(filename)
                    )
                

        return x_training_data, y_training_data


    @benchmark
    def train_ann(self, x_training_data: list, y_training_data: list) -> None:
        self.__ann_model.train_model(x_training_data, y_training_data)


    @benchmark
    def train_3d_cnn(self, x_training_data: list, y_training_data: list) -> None:
        self.__3d_cnn_model.train_model(x_training_data, y_training_data)


    @benchmark
    def train_cnn(self, x_training_data: list, y_training_data: list) -> None:
        self.__cnn_model.train_model(x_training_data, y_training_data)


    @benchmark
    def save_ann(self) -> None:
        self.__ann_model.save_model()


    @benchmark
    def save_3d_cnn(self) -> None:
        self.__3d_cnn_model.save_model()


    @benchmark
    def save_cnn(self) -> None:
        self.__cnn_model.save_model()

    
    @benchmark
    def evaluate_cnn_image(self, image: str) -> Tuple[str, list]:
        return self.__cnn_model.evaluate_image(image)


    @benchmark
    def evaluate_ann_video(self, video: str) -> Tuple[str, list]:
        return self.__ann_model.evaluate_video(video)


    @benchmark
    def evaluate_3d_cnn_video(self, video: str) -> Tuple[str, list]:
        return self.__3d_cnn_model.evaluate_video(video)
