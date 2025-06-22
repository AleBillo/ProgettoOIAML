import cv2
import numpy as np

# import mediapipe
# 
# def preprocess_mediapipe(img, target_size=(50, 50), color_threshold=50, smoothing_ksize=3):
#     with mediapipe.solutions.hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
#         results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# 
#     mask = np.zeros(img.shape[:2], dtype=np.uint8)
# 
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             points = [(int(l.x * img.shape[1]), int(l.y * img.shape[0])) for l in hand_landmarks.landmark]
#             hull = cv2.convexHull(np.array(points))
#             cv2.drawContours(mask, [hull], -1, 255, -1)
# 
#         y_coords, x_coords = np.where(mask == 255)
#         if len(x_coords) == 0 or len(y_coords) == 0:
#             pass
#         else:
#             x_min, x_max = x_coords.min(), x_coords.max()
#             y_min, y_max = y_coords.min(), y_coords.max()
# 
#             roi_img = img[y_min:y_max+1, x_min:x_max+1]
#             roi_mask_for_refinement = mask[y_min:y_max+1, x_min:x_max+1]
# 
#             roi_pixels = roi_img[roi_mask_for_refinement == 255]
#             if roi_pixels.size == 0:
#                 pass
#             else:
#                 avg_color = np.mean(roi_pixels, axis=0)
#                 roi_img_float = roi_img.astype(np.float32)
#                 dist_map = np.sqrt(np.sum((roi_img_float - avg_color)**2, axis=2))
#                 roi_mask_for_refinement[dist_map > color_threshold] = 0
#                 mask[y_min:y_max+1, x_min:x_max+1] = roi_mask_for_refinement
# 
#     if smoothing_ksize > 1:
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smoothing_ksize, smoothing_ksize))
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# 
#     mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_AREA)
#     mask = np.expand_dims(mask, axis=0)
#     return mask

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

def preprocess_greyscale(img, target_size=(50, 50)):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, target_size)
    gray = np.expand_dims(gray, axis=0)
    return gray

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

preprocess = preprocess_greyscale
