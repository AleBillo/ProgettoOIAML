import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        base_model = models.resnet18(weights=None)
        base_model.conv1.in_channels = 1
        base_model.conv1.weight = nn.Parameter(
                base_model.conv1.weight.sum(1, keepdim=True)
                )
        in_feats = base_model.fc.in_features
        base_model.fc = nn.Linear(in_feats, num_classes)
        self.model = base_model

    def forward(self, x):
        return self.model(x)
