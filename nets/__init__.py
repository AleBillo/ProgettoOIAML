from .cnn import CNN
from .resnet import ResNet

def get_model(model_name, **kwargs):
    mapping = {
            "CNN": CNN,
            "ResNet": ResNet
            }
    if model_name in mapping:
        return mapping[model_name](**kwargs)
    raise ValueError(f"Unknown model: {model_name}")
