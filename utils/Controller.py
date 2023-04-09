from data_generators.TrainingDataGeneratorANN import TrainingDataGeneratorANN
from utils.decorators import benchmark
import os

class ControllerClass:
    def __init__(
        self, 
        detection_confidence = 0.3, 
        tracking_confidence = 0.3,
        complexity = 1
    ) -> None:
        self.__detection_confidence = detection_confidence
        self.__tracking_confidence = tracking_confidence
        self.__complexity = complexity


    @benchmark
    def gather_data_for_ann(self):
        counter_files = 1
        training_data = []
        
        for filename in os.listdir(os.getcwd()  + "/training data"):
            print(f"Processing file {counter_files} with the name {filename}")
            counter_files+=1
            trainer = TrainingDataGeneratorANN(
                vid=filename,
                detection_confidence=self.__detection_confidence, 
                tracking_confidence=self.__tracking_confidence, 
                complexity=self.__complexity,
                )
            final_values_vector = trainer.refactor_data()
            final_values_vector = final_values_vector[0:230]
            training_data.append(
                (
                    final_values_vector,
                    filename.split("_")[0]
                )
            )
            
        return training_data
