import os
import tensorflow as tf


class MobileNetV2:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load(self):
        if self.model_path and os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            print("Model path not provided or model does not exist.")

    def predict(self, input_data):
        return self.model.predict(input_data)
