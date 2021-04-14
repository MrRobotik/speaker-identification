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


# CALCULATING AVG SESSION XVECTORS FOR EVERY SPEAKER IN TESTING DATASET (functions used only in jupyter notebook)

# def get_avg_xvector(model, dataloader, device):
#     avg_xvectors = torch.Tensor().to(device)
#
#     with torch.no_grad():
#         for inputs, _ in dataloader:
#             inputs = inputs.to(device)
#             xvectors, _ = model(inputs)
#             avg_xvectors = torch.cat((avg_xvectors, torch.mean(xvectors, 0).view(1, 512)), 0)
#
#     return torch.mean(avg_xvectors, 0).tolist()
#
#
# def get_session_xvectors(model, data_folder, chunk_size, batch_size, device):
#     session_xvectors = []
#     speaker_labels = []
#     speaker_label = 0
#
#     for speaker_folder in listdir(data_folder):
#         for session_folder in listdir(data_folder / speaker_folder):
#
#             datasets_list = []
#             for features_file in listdir(data_folder / speaker_folder / session_folder):
#                 if features_file.split('-')[1] == "mfcc.npy":
#                     mfcc_file = data_folder / speaker_folder / session_folder / features_file
#                     datasets_list.append(MFCCDataset(mfcc_file, speaker_label, chunk_size))
#
#             test_dataset = ConcatDataset(datasets_list)
#             test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)
#             session_xvectors.append(get_avg_xvector(model, test_dataloader, device))
#             speaker_labels.append(speaker_label)
#
#         speaker_label += 1
#
#     return session_xvectors, speaker_labels
