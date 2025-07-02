from .blob import preprocess_hsv_largest_blob_filled
from .grayscale import preprocess_greyscale
from .hsv import preprocess_hsv
# from .mediapipe import preprocess_mediapipe

def get_preprocessor(method="greyscale"):
    mapping = {
            "greyscale": preprocess_greyscale,
            "hsv": preprocess_hsv,
            "blob": preprocess_hsv_largest_blob_filled,
            # "mediapipe": preprocess_mediapipe
            }
    if method in mapping:
        return mapping[method]
    raise ValueError(f"Unknown preprocessing method: {method}")
