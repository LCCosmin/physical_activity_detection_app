from enum import Enum

class TrainerEnum(Enum):
    ANN = 'ANN'
    CNN = 'CNN'
    CNN_3D = 'CNN_3D'

class TrainerAction(Enum):
    TRAIN = 'TRAIN'
    EVALUATE = 'EVALUATE'
