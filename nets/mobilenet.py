import torch.nn as nn
import torchvision.models as models

class MobileNet(nn.Module):
    def __init__(self, num_classes=3, input_size=None):
        super().__init__()
        base_model = models.mobilenet_v2(weights=False)

        base_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        in_feats = base_model.last_channel
        base_model.classifier[1] = nn.Linear(in_feats, num_classes)
        self.model = base_model

    def forward(self, x):
        return self.model(x)
