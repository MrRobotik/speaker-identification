import torch
import torch.nn as nn
import torch.nn.functional as F


class XVectors(nn.Module):

    def __init__(self):
        super().__init__()
        self.tdnn = nn.Sequential(
            nn.Conv1d(in_channels=24,  out_channels=512,  kernel_size=5, dilation=1), nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512,  kernel_size=3, dilation=2), nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512,  kernel_size=3, dilation=3), nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512,  kernel_size=1, dilation=1), nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1), nn.ReLU()
        )
        self.bn1 = nn.BatchNorm1d(3000)
        self.fc1 = nn.Linear(3000, 512)

    def forward(self, utters):
        stats_pools = []
        for x in utters:
            y = torch.squeeze(self.tdnn(x))
            stats_pool = torch.cat((y.mean(dim=1), y.std(dim=1)), dim=0)
            stats_pools.append(stats_pool)

        x = torch.stack(stats_pools, dim=0)
        embed_a = F.relu(self.fc1(self.bn1(x)))
        return embed_a

    @staticmethod
    def min_sample_length():
        return 15  # samples shorter than 0.225 second will crash on TDNN layer
