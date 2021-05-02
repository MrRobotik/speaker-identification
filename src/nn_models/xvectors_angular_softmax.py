import torch.nn as nn
import torch.nn.functional as F
from .angular_softmax import AngularSoftmax
from .xvectors import XVectors


class XVectorsAngularSoftmax(XVectors):

    def __init__(self, output_classes=1211):
        super().__init__()
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.angular_softmax = AngularSoftmax(512, output_classes)

    def forward(self, utters, labels=None, training=True):
        embed_a = super().forward(utters)

        if training:
            embed_b = self.fc2(self.bn2(F.relu(embed_a)))
            loss = self.angular_softmax(embed_b, labels)
            return loss
        else:
            return embed_a
