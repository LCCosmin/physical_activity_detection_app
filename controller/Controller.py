from dataclasses import dataclass
from typing import Union
from data_generators.TrainingDataGeneratorANN import TrainingDataGeneratorANN
from data_generators.TrainingDataGenerator3DCNN import TrainingDataGenerator3DCNN
from utils.decorators import benchmark
from models.VectorModelANN import VectorModelANN
import os
from utils.utils import transfor_file_name_into_int


@dataclass
class ControllerClass:
    __ann_model: VectorModelANN
    __ann_detection_confidence_gen_ann: float
    __ann_tracking_confidence_gen_ann: float
    __ann_complexity_gen_ann: int
    __3d_cnn_width: int
    __3d_cnn_height: int


    def __init__(
        self, 
        detection_confidence_gen_ann, 
        tracking_confidence_gen_ann,
        complexity_gen_ann,
        width_3d_cnn,
        height_3d_cnn
    ) -> None:
        self.__ann_detection_confidence_gen_ann = detection_confidence_gen_ann
        self.__ann_tracking_confidence_gen_ann = tracking_confidence_gen_ann
        self.__ann_complexity_gen_ann = complexity_gen_ann
        self.__3d_cnn_width = width_3d_cnn
        self.__3d_cnn_height = height_3d_cnn

        self.__ann_model = VectorModelANN()


    @benchmark
    def gather_data_for_ann(self) -> Union[list, list]:
        counter_files = 1
        x_training_data = []
        y_training_data = []
        
        for filename in os.listdir(os.getcwd()  + "/training data"):
            print(f"INFO:GATHER_DATA_ANN: Processing file {counter_files} with the name {filename}")
            counter_files+=1
            
            trainer = TrainingDataGeneratorANN(
                vid=filename,
                detection_confidence=self.__ann_detection_confidence_gen_ann, 
                tracking_confidence=self.__ann_tracking_confidence_gen_ann, 
                complexity=self.__ann_complexity_gen_ann,
                )
            
            vector_values = trainer.generate_vector_data()

            x_training_data.append(
                vector_values
            )
            y_training_data.append(
                transfor_file_name_into_int(filename)
            )           
            
        return x_training_data, y_training_data

    # @Cosmin Must refactor this shit
    @benchmark
    def gather_data_for_3d_cnn(self) -> Union[list, list]:
        x_training_data = []
        y_training_data = []
        counter_files = 1

        for filename in os.listdir(os.getcwd()  + "/training data"):
            print(f"INFO:GATHER_DATA_3D_CNN: Processing file {counter_files} with the name {filename}")
            counter_files+=1

            trainer = TrainingDataGenerator3DCNN(
                vid=filename,
                cnn_3d_width=self.__3d_cnn_width,
                cnn_3d_height=self.__3d_cnn_height,
            )

            vector_values_3d = trainer.generate_3d_data()

            x_training_data.append(
                vector_values_3d
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
