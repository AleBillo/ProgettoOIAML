import cv2
import numpy as np

def preprocess_hsv(img, target_size=(50, 50)):
    if img is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV range for skin
    lower = np.array([2, 50, 60], dtype=np.uint8)
    upper = np.array([25, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)  # white = skin, black = background

    # Clean up mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Binarize explicitly
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Resize
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    # Add channel dimension
    mask = np.expand_dims(mask, axis=0)

    return mask

