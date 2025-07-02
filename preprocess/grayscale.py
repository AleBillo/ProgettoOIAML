import cv2
import numpy as np

def preprocess_greyscale(img, target_size=(50, 50)):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, target_size)
    gray = np.expand_dims(gray, axis=0)
    return gray
