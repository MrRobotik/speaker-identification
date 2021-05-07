import torch.nn as nn
import torch.nn.functional as F
from .angular_softmax import AngularSoftmax
from .xvectors import XVectors


class XVectorsAngularSoftmax(XVectors):

    def __init__(self, output_classes=1211):
        super().__init__()
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.angular_softmax = AngularSoftmax(512, output_classes)

    def forward(self, utters, labels=None):
        embed_a = super().forward(utters)
        embed_b = F.relu(self.fc2(self.bn2(embed_a)))
        loss = self.angular_softmax(self.bn3(embed_b), labels)
        return loss
