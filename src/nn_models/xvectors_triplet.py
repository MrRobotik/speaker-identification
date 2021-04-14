from . xvectors import XVectors


class XVectorsTriplet(XVectors):

    def __init__(self):
        super().__init__()

    def forward(self, utters):
        embed_a = super().forward(utters)
        return embed_a
