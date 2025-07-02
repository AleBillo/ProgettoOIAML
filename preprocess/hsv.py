import cv2
import numpy as np

def preprocess_hsv(img, target_size=(50, 50)):
    if img is None:
        return None

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV skin color range (tune as needed for your lighting)
    lower = np.array([2, 50, 60], dtype='uint8')
    upper = np.array([25, 150, 255], dtype='uint8')
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological operations to clean up
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Threshold to binary (optional, but ensures only 0/255 values)
    _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)

    # Resize to target_size
    mask = cv2.resize(mask, target_size)

    # Add channel dimension to match shape (1, H, W)
    mask = np.expand_dims(mask, axis=0)
    return mask
