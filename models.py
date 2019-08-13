from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet



class ResNet18(nn.Module):
    def __init__(self, n_classes=2, pretrained=True):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        features_num = self.model.fc.in_features
        self.model.fc = nn.Linear(features_num, n_classes)

    def forward(self, x):
        return self.model(x)

class ResNet50(nn.Module):
    def __init__(self, n_classes=2, pretrained=True):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        features_num = self.model.fc.in_features
        self.model.fc = nn.Linear(features_num, n_classes)

    def forward(self, x):
        return self.model(x)

class EfficientNetB0(nn.Module):

    def __init__(self, n_classes=2, pretrained=True)
        super(EfficientNetB0, self).__init__()
        if pretrained:
            self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=n_classes)
        else:
            self.model = EfficientNet.from_name('efficientnet-b0')

    def forward(self, x):
        return self.model(x)