import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import time
import numpy as np
from os import listdir
from pathlib import Path


# TRAINING DATASET CONSTRUCTION:

class MFCCDataset(Dataset):

    def __init__(self, mfcc_file, speaker_label, chunk_size=24):
        self.chunk_size = chunk_size
        self.speaker_label = speaker_label
        loaded_mfcc = np.load(mfcc_file).T
        self.mfcc_count = int(loaded_mfcc.shape[0] / chunk_size)
        self.mfcc = torch.from_numpy(loaded_mfcc[:self.mfcc_count * self.chunk_size, :])
        self.attr_len = self.mfcc.shape[1]

    def __len__(self):
        return self.mfcc_count

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = start_idx + self.chunk_size
        truncated_mfcc = self.mfcc[start_idx:end_idx, :]
        return truncated_mfcc.view(self.chunk_size, self.attr_len), self.speaker_label


def create_training_mfcc_dataset(data_folder, chunk_size=24):
    datasets_list = []
    speaker_label = -1

    for speaker_folder in listdir(data_folder):
        speaker_label += 1
        for session_folder in listdir(data_folder / speaker_folder):
            for features_file in listdir(data_folder / speaker_folder / session_folder):

                if features_file.split('-')[1] == "mfcc.npy":
                    mfcc_file = data_folder / speaker_folder / session_folder / features_file
                    datasets_list.append(MFCCDataset(mfcc_file, speaker_label, chunk_size))

    return ConcatDataset(datasets_list)


# MODEL DEFINITION:

class XVectorsBaseline(nn.Module):

    def __init__(self, output_classes, chunk_size=24):
        super(XVectorsBaseline, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=chunk_size, out_channels=512, kernel_size=5, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, dilation=2)
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, dilation=3)
        self.conv4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)

        self.fc1 = nn.Linear(3000, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # stats pooling
        x = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)

        xvectors = self.fc1(x)
        x = F.relu(xvectors)
        x = F.relu(self.fc2(x))

        return xvectors, self.fc3(x)


# TRAINING MODEL:

def compute_accuracy(model, dataloader, device):
    n_samples = 0
    n_correct = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            _, outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predicted == labels).sum().item()

    return round(n_correct / n_samples * 100, 3)


def train_model(model, params_location, epochs_num, train_dataloader, optimizer, criterion, device):
    for epoch in range(epochs_num):
        epoch_loss = 0.0
        start_time = time.time()

        for mb, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        end_time = time.time()

        accuracy = compute_accuracy(model, train_dataloader, device)
        print(f"Epoch {epoch}: loss: {epoch_loss}, accuracy: {accuracy}, duration: {round(end_time - start_time, 3)}s")

        torch.save(model.state_dict(), params_location)
        print(f"Model params were resaved!\n")


# CALCULATING AVG SESSION XVECTORS FROM TRAINED MODEL (this functions are used only in jupyter notebook):

def get_avg_xvector(model, dataloader, device):
    avg_xvectors = torch.Tensor().to(device)

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            xvectors, _ = model(inputs)
            avg_xvectors = torch.cat((avg_xvectors, torch.mean(xvectors, 0).view(1, 512)), 0)

    return torch.mean(avg_xvectors, 0).tolist()


def get_session_xvectors(model, data_folder, chunk_size, batch_size, device):
    xvectors = []
    labels = []
    speaker_label = 0

    for speaker_folder in listdir(data_folder):
        for session_folder in listdir(data_folder / speaker_folder):

            datasets_list = []
            for features_file in listdir(data_folder / speaker_folder / session_folder):
                if features_file.split('-')[1] == "mfcc.npy":
                    mfcc_file = data_folder / speaker_folder / session_folder / features_file
                    datasets_list.append(MFCCDataset(mfcc_file, speaker_label, chunk_size))

            test_dataset = ConcatDataset(datasets_list)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)
            xvectors.append(get_avg_xvector(model, test_dataloader, device))
            labels.append(speaker_label)

        speaker_label += 1

    return xvectors, labels


# MAIN:

def main():
    # training configuration:
    training_data = Path("/home/joey/School/KNN/speaker-identification/data/vox1_dev/")
    model_params = Path("/home/joey/School/KNN/speaker-identification/model_params.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    batch_size = 2048
    chunk_size = 24

    # model declaration:
    speakers_count = len(listdir(training_data))
    model = XVectorsBaseline(speakers_count, chunk_size).to(device)

    if model_params.is_file():
        model.load_state_dict(torch.load(model_params))
        print("Model parameters were loaded!")

    # loss and optimizer selection:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # create training dataset and dataloader:
    train_dataset = create_training_mfcc_dataset(training_data, chunk_size)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # calling train model function:
    train_model(model, model_params, epochs, train_dataloader, optimizer, criterion, device)


if __name__ == '__main__':
    main()
