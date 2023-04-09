from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


@dataclass(kw_only=True)
class PictureModelCNN:
    _width_crop: int = 20
    _height_crop: int = 60
    _epochs_no: int = 256
    _batch_size: int = 32
    _checkpoint_path: str = field(init=False)
    _training_folder: str = field(init=False)

    def __post_init__(self) -> None:
        self._training_folder = "./training_data_picture_model_cnn"
        self._checkpoint_path = "./brains/picture_model_cnn/cp.ckpt"
    
    def normalize(self, img):
        img = cv2.resize(img, (self._width_crop, self._height_crop), interpolation = cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (1,1), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255

        return img

    def create_model(self):
        model = tf.keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(self._width_crop * self._height_crop,)),
            keras.layers.Dense(400, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(200, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(100, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(50, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            
            keras.layers.Dense(2, activation = 'sigmoid')
            ])
        
        model.compile(optimizer = 'adam', 
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics = ['accuracy'])

        return model

    def gather_training_data(self):
        print("Gather all data for training ...")
        x_load_images = []
        y_load_images = []
        for filename in os.listdir(self._training_folder):
            #Read one image from folder
            img = cv2.imread(os.path.join(self._training_folder,filename))
            if img is not None:
                image = self.normalize(img)
                x_load_images.append(image)

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
                    y_load_images.append(0)
                elif "back" in filename:
                    y_load_images.append(1)
                elif "biceps" in filename:
                    y_load_images.append(2)
                elif "butt" in filename:
                    y_load_images.append(3)
                elif "chest" in filename:
                    y_load_images.append(4)
                elif "forearm" in filename:
                    y_load_images.append(5)
                elif "legs" in filename:
                    y_load_images.append(6)
                elif "shoulder" in filename:
                    y_load_images.append(7)
                else:
                    y_load_images.append(8)

        print("For training were used:")
        print("{} pictures with abs exercise".format(y_load_images.count(0)))
        print("{} pictures with abs exercise".format(y_load_images.count(1)))
        print("{} pictures with abs exercise".format(y_load_images.count(2)))
        print("{} pictures with abs exercise".format(y_load_images.count(3)))
        print("{} pictures with abs exercise".format(y_load_images.count(4)))
        print("{} pictures with abs exercise".format(y_load_images.count(5)))
        print("{} pictures with abs exercise".format(y_load_images.count(6)))
        print("{} pictures with abs exercise".format(y_load_images.count(7)))
        print("{} pictures with abs exercise".format(y_load_images.count(8)))
        return (x_load_images, y_load_images)

    def train_and_save(self) -> None:
        #Get the training data
        print("Starting the training protocol ...")
        x_load_images = []
        y_load_images = []

        x_load_images, y_load_images = self.gather_training_data()

        print ("Start training ...")
        x_personnel = np.array(x_load_images, dtype=(float)).reshape(-1, self._width_crop * self._height_crop)
        y_personnel = np.array(y_load_images, dtype=(int))

        x_train, x_test, y_train, y_test = train_test_split(x_personnel, y_personnel, 
                                                            test_size = 0.2, shuffle=(True))

        model = self.create_model()

        history = model.fit(x_train, y_train, epochs = self._epochs_no, batch_size = self._batch_size)
        model.save_weights(self._checkpoint_path)

        loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

        print(model.predict(x_test)[0])

        plt.plot(history.history['loss'])
        plt.show(block=True)
        plt.savefig("./loss.png")

        print("Accuracy : " + str(accuracy))

        print("Training done ..")

    def evaluate_frames(self, people_list) -> None:
        ...

    def evaluate_image(self, image) -> None:
        ...