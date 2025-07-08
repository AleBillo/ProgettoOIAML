import torch.nn as nn
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self, num_classes=3, input_size=None):
        super().__init__()
        
        base_model = models.vgg11_bn(weights=False)
        
        base_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        
        in_feats = base_model.classifier[-1].in_features
        base_model.classifier[-1] = nn.Linear(in_feats, num_classes)
        self.model = base_model

    def forward(self, x):
        return self.model(x)
