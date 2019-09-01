from torch import nn
from torchvision import models
import torch

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

class ShuffleNetv2(nn.Module):

    def __init__(self, n_classes=2, pretrained=True):
        super(ShuffleNetv2, self).__init__()
        self.model =  models.shufflenet_v2_x1_0(pretrained=pretrained)
        features_num = self.model.fc.in_features
        self.model.fc = nn.Linear(features_num, n_classes)

    def forward(self, x):
        return self.model(x)


class LSTMClassification(nn.Module):
    def __init__(self, n_classes=3):
        super(LSTMClassification, self).__init__()
        # backbone = models.resnet34(pretrained=True)
        backbone = models.shufflenet_v2_x1_0(pretrained=True)
        layers = list(backbone.children())
        self.backbone = nn.Sequential(*layers[:-1])
        # self.backbone = backbone.features
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_dim = layers[-1].in_features
        self.lstm = nn.LSTM(self.embedding_dim, 512)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        preds = []

        for i, segment_images in enumerate(x):
            segment_features = self.backbone(segment_images)
            # segment_features = torch.flatten(segment_features, 1)
            # segment_features = segment_features
            # segment_features = self.pool(segment_features)
            segment_features = segment_features.mean([2, 3])
            lstm_features, _ = self.lstm(segment_features.view(len(segment_features), 1, self.embedding_dim))
            res = self.fc(lstm_features[-1])
            preds.append(res)

        return torch.cat(preds)


class LSTMDeepClassification(nn.Module):
    def __init__(self, n_classes=3):
        super(LSTMDeepClassification, self).__init__()
        # backbone = models.resnet34(pretrained=True)
        backbone = models.shufflenet_v2_x1_0(pretrained=True)
        layers = list(backbone.children())
        self.backbone = nn.Sequential(*layers[:-1])
        # self.backbone = backbone.features
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_dim = layers[-1].in_features
        self.lstm = nn.LSTM(self.embedding_dim, 512, num_layers=2)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        preds = []

        for i, segment_images in enumerate(x):
            segment_features = self.backbone(segment_images)
            # segment_features = torch.flatten(segment_features, 1)
            # segment_features = segment_features
            # segment_features = self.pool(segment_features)
            segment_features = segment_features.mean([2, 3])
            lstm_features, _ = self.lstm(segment_features.view(len(segment_features), 1, self.embedding_dim))
            res = self.fc(lstm_features[-1])
            preds.append(res)

        return torch.cat(preds)