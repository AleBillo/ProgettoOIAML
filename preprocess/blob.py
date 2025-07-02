import cv2
import numpy as np

def preprocess_hsv_largest_blob_filled(img, target_size=(50, 50)):
    if img is None:
        return None

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV skin color range (adjust as needed)
    lower = np.array([2, 50, 60], dtype='uint8')
    upper = np.array([25, 150, 255], dtype='uint8')
    skin_mask = cv2.inRange(hsv, lower, upper)

    # Morphological operations to clean up
    skin_mask = cv2.GaussianBlur(skin_mask, (5,5), 0)
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # Threshold to binary
    _, skin_mask = cv2.threshold(skin_mask, 30, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # No skin-like area detected
        mask = np.zeros(target_size, dtype=np.uint8)
        mask = np.expand_dims(mask, axis=0)
        return mask

    # Find the largest contour (by area)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a blank mask
    filled_mask = np.zeros_like(skin_mask)

    # Draw and fill the largest contour
    cv2.drawContours(filled_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Resize
    filled_mask = cv2.resize(filled_mask, target_size)

    # Add channel dimension
    filled_mask = np.expand_dims(filled_mask, axis=0)
    return filled_mask
