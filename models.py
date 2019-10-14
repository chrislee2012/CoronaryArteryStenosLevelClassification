from torch import nn
from torchvision import models
import torch
# from pretrainedmodels import se_resnext50_32x4d
import torch.nn.functional as F


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
        self.model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        features_num = self.model.fc.in_features
        self.model.fc = nn.Linear(features_num, n_classes)

    def forward(self, x):
        return self.model(x)

# class SeResNext50(nn.Module):
#
#     def __init__(self, n_classes=2, pretrained=True):
#         super(SeResNext50, self).__init__()
#         if pretrained:
#             self.model = se_resnext50_32x4d(pretrained='imagenet')
#         else:
#             self.model = se_resnext50_32x4d()
#         features_num = 204800#self.model.last_linear.in_features
#         self.model.last_linear = nn.Linear(features_num, n_classes)
#
#     def forward(self, x):
#         return self.model(x)

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


class LSTMDeepResNetClassification(nn.Module):
    def __init__(self, n_classes=3):
        super(LSTMDeepResNetClassification, self).__init__()
        backbone = models.resnet34(pretrained=True)
        layers = list(backbone.children())
        self.backbone = nn.Sequential(*layers[:-1])
        self.embedding_dim = layers[-1].in_features
        self.lstm = nn.LSTM(self.embedding_dim, 512, num_layers=2)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        preds = []

        for i, segment_images in enumerate(x):
            segment_features = self.backbone(segment_images)
            segment_features = segment_features.mean([2, 3])
            lstm_features, _ = self.lstm(segment_features.view(len(segment_features), 1, self.embedding_dim))
            res = self.fc(lstm_features[-1])
            preds.append(res)

        return torch.cat(preds)


class AttentionResNet18(nn.Module):
    def __init__(self, n_classes=3, pretrained=True):
        super(AttentionResNet18, self).__init__()
        self.M = 512
        self.L = 512

        backbone = models.resnet18(pretrained=pretrained)
        layers = list(backbone.children())
        extractor = nn.Sequential(*layers[:-1])

        self.feature_extractor_part1 = extractor

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L, bias=False),
            nn.Tanh(),
            nn.Linear(self.L, 1, bias=False),
            nn.Softmax(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, n_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        features = self.feature_extractor_part1(x)
        features = features.view(batch_size, features.size(0), -1)

        attention = self.attention(features)
        out = (attention * features).sum(1)
        out = self.classifier(out)

        return out


class AttentionResNet34(nn.Module):
    def __init__(self, n_classes=3, pretrained=True):
        super(AttentionResNet34, self).__init__()
        self.L = 512

        backbone = models.resnet34(pretrained=pretrained)
        self.M = backbone.fc.in_features
        layers = list(backbone.children())
        extractor = nn.Sequential(*layers[:-1])

        self.feature_extractor_part1 = extractor

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L, bias=False),
            nn.Tanh(),
            nn.Linear(self.L, 1, bias=False),
            nn.Softmax(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, n_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        features = self.feature_extractor_part1(x)
        features = features.view(batch_size, features.size(0), -1)

        attention = self.attention(features)
        out = (attention * features).sum(1)
        out = self.classifier(out)

        return out


class AttentionShuffleNetV2(nn.Module):
    def __init__(self, n_classes=3, pretrained=True):
        super(AttentionShuffleNetV2, self).__init__()
        self.L = 512

        backbone = models.shufflenet_v2_x1_0(pretrained=pretrained)
        self.M = backbone.fc.in_features * 256
        layers = list(backbone.children())
        extractor = nn.Sequential(*layers[:-1])

        self.feature_extractor_part1 = extractor

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L, bias=False),
            nn.Tanh(),
            nn.Linear(self.L, 1, bias=False),
            nn.Softmax(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, n_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        features = self.feature_extractor_part1(x)

        features = features.view(features.size(0), features.size(1) * features.size(2) * features.size(3))
        features = features.view(batch_size, features.size(0), -1)

        attention = self.attention(features)
        out = (attention * features).sum(1)
        out = self.classifier(out)

        return out


class AttentionSqueezeNet(nn.Module):
    def __init__(self, n_classes=3, pretrained=True):
        super(AttentionSqueezeNet, self).__init__()
        self.L = 512

        backbone = models.squeezenet1_1(pretrained=pretrained)
        layers = list(backbone.children())
        self.M = 492032
        extractor = nn.Sequential(*layers[:-1])

        self.feature_extractor_part1 = extractor

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L, bias=False),
            nn.Tanh(),
            nn.Linear(self.L, 1, bias=False),
            nn.Softmax(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, n_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        features = self.feature_extractor_part1(x)

        features = features.view(features.size(0), features.size(1) * features.size(2) * features.size(3))
        features = features.view(batch_size, features.size(0), -1)

        attention = self.attention(features)
        out = (attention * features).sum(1)
        out = self.classifier(out)

        return out

