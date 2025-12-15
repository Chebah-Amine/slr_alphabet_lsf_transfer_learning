import cv2
import numpy as np


def image_treatment(image, dimension):
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img
