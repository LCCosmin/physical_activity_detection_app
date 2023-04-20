from dataclasses import dataclass


@dataclass(kw_only=True)
class TrainerANNData:
    detection_confidence: float
    tracking_confidence: float
    complexity: int


    def __init__(
        self,
        detection_confidece,
        tracking_confidence,
        complexity,
    ) -> None:
        self.detection_confidence = detection_confidece
        self.tracking_confidence = tracking_confidence
        self.complexity = complexity
        

@dataclass(kw_only=True)
class Trainer3DCNNData:
    width_3d_cnn: int
    height_3d_cnn: int


    def __init__(self, width_3d_cnn, height_3d_cnn) -> None:
        self.width_3d_cnn = width_3d_cnn
        self.height_3d_cnn = height_3d_cnn


@dataclass(kw_only=True)
class TrainerCNNData:
    width_cnn: int
    height_cnn: int


    def __init__(self, width_cnn, height_cnn) -> None:
        self.width_cnn = width_cnn
        self.height_cnn = height_cnn
