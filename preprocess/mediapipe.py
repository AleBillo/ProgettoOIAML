import mediapipe
import cv2
import numpy as np

def preprocess_mediapipe(img, target_size=(50, 50), color_threshold=50, smoothing_ksize=3):
    with mediapipe.solutions.hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            points = [(int(l.x * img.shape[1]), int(l.y * img.shape[0])) for l in hand_landmarks.landmark]
            hull = cv2.convexHull(np.array(points))
            cv2.drawContours(mask, [hull], -1, 255, -1)

        y_coords, x_coords = np.where(mask == 255)
        if len(x_coords) == 0 or len(y_coords) == 0:
            pass
        else:
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            roi_img = img[y_min:y_max+1, x_min:x_max+1]
            roi_mask_for_refinement = mask[y_min:y_max+1, x_min:x_max+1]

            roi_pixels = roi_img[roi_mask_for_refinement == 255]
            if roi_pixels.size == 0:
                pass
            else:
                avg_color = np.mean(roi_pixels, axis=0)
                roi_img_float = roi_img.astype(np.float32)
                dist_map = np.sqrt(np.sum((roi_img_float - avg_color)**2, axis=2))
                roi_mask_for_refinement[dist_map > color_threshold] = 0
                mask[y_min:y_max+1, x_min:x_max+1] = roi_mask_for_refinement

    if smoothing_ksize > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smoothing_ksize, smoothing_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_AREA)
    mask = np.expand_dims(mask, axis=0)
    return mask
