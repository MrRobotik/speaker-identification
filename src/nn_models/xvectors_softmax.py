import torch.nn as nn
import torch.nn.functional as F
from . xvectors import XVectors


class XVectorsSoftmax(XVectors):

    def __init__(self, output_classes=1211):
        super().__init__()
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, output_classes)

    def forward(self, utters):
        embed_a = super().forward(utters)
        embed_b = self.fc2(self.bn2(F.relu(embed_a)))
        logits  = self.fc3(self.bn3(F.relu(embed_b)))
        return logits
