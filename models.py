from torch import nn
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, n_classes=2, pretrained=True):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        features_num = self.model.fc.in_features
        self.model.fc = nn.Linear(features_num, n_classes)

    def forward(self, x):
        return self.model(x)
