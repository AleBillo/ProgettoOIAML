from .cnn import CNN
from .resnet import ResNet
from .vgg import VGG
from .mobilenet import MobileNet

def get_model(model_name, **kwargs):
    mapping = {
        "CNN": CNN,
        "ResNet": ResNet,
        "VGG": VGG,
        "MobileNet": MobileNet
    }
    if model_name in mapping:
        return mapping[model_name](**kwargs)
    raise ValueError(f"Unknown model: {model_name}")
