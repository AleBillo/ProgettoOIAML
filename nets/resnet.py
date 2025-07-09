import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes=3, input_size=None):
        super().__init__()
        base_model = models.resnet18(weights=None)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_feats = base_model.fc.in_features
        base_model.fc = nn.Linear(in_feats, num_classes)
        self.model = base_model

    def forward(self, x):
        return self.model(x)
