from data_generators.TrainingDataGeneratorANN import TrainingDataGeneratorANN
from utils.decorators import benchmark
from models.VectorModelANN import VectorModelANN
import os


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


def extract_data_from_video_for_ann_prediction(video):
    trainer = TrainingDataGeneratorANN(
            vid=video
        )

    vector_data = trainer.generate_vector_data()

    return vector_data


class ControllerClass:
    __ann_model: VectorModelANN

    def __init__(
        self, 
        detection_confidence = 0.3, 
        tracking_confidence = 0.3,
        complexity = 1
    ) -> None:
        self.__ann_detection_confidence = detection_confidence
        self.__ann_tracking_confidence = tracking_confidence
        self.__ann_complexity = complexity


    @benchmark
    def gather_data_for_ann(self):
        counter_files = 1
        x_training_data = []
        y_training_data = []
        
        for filename in os.listdir(os.getcwd()  + "/training data"):
            print(f"INFO:GATHER_DATA_ANN: Processing file {counter_files} with the name {filename}")
            counter_files+=1
            
            trainer = TrainingDataGeneratorANN(
                vid=filename,
                detection_confidence=self.__ann_detection_confidence, 
                tracking_confidence=self.__ann_tracking_confidence, 
                complexity=self.__ann_complexity,
                )
            
            final_values_vector = trainer.generate_vector_data()
            final_values_vector = final_values_vector[0:230]

            x_training_data.append(
                final_values_vector
            )
            y_training_data.append(
                transfor_file_name_into_int(filename)
            )           
            
        return x_training_data, y_training_data


    @benchmark
    def train_ann(self):
        x_train, y_train = self.gather_data_for_ann()
        
        self.__ann_model.train_and_save(x_train, y_train)


