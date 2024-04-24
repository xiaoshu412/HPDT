import torch
import torch.nn as nn
from torchvision.models import densenet121
class Encoder(nn.Module):
    def __init__(self, features_size=2048, dropout_prob=0.5):
        super(Encoder, self).__init__()
        self.densenet = densenet121(pretrained=True)
        self.densenet.classifier = nn.Sequential(
            nn.Linear(self.densenet.classifier.in_features, features_size),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.densenet(x)
        x = self.dropout(x)
        return x

class Classifier(nn.Module):
    def __init__(self, features_size=2048, n_classes=7):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(features_size, n_classes)


    def forward(self, x):
        x = self.fc(x)
        return x
