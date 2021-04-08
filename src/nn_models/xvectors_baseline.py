import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import ceil
from os import listdir
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split


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
        return self.mfcc[start_idx:end_idx, :], self.speaker_label


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


def processing_run(best_accuracy, device, epoch, epochs, model, params_loc, run_id, run_loss, run_processed_mb,
                   run_start, runs_num, val_dataloader):
    run_end = time.time()
    run_accuracy = compute_accuracy(model, val_dataloader, device)
    run_duration = round(run_end - run_start, 3)

    print(f"epoch: {epoch}/{epochs}, "
          f"run: {run_id}/{runs_num}, "
          f"processed m.b.: {run_processed_mb}, "
          f"avg.m.b. loss:  {round(run_loss / run_processed_mb, 3)}, "
          f"accuracy: {run_accuracy}%, "
          f"duration: {run_duration}s")

    if run_accuracy > best_accuracy:
        increase = round(run_accuracy - best_accuracy, 3)
        best_accuracy = run_accuracy
        torch.save(model.state_dict(), params_loc)
        print(f"Accuracy on validation set has been increased by {increase}%. Model parameters were resaved!")

    return best_accuracy


def train_model(model, params_loc, epochs, run_mb_num, train_dataloader, val_dataloader, optimizer, criterion, device):
    runs_num = max(ceil(len(train_dataloader) / run_mb_num), 1)
    best_accuracy = compute_accuracy(model, val_dataloader, device)
    print(f"Initial accuracy on validation set: {best_accuracy}%")

    for epoch in range(1, epochs + 1, 1):
        run_id = 1
        run_processed_mb = 0
        epoch_loss = run_loss = 0.0
        epoch_start = run_start = time.time()

        for mb, (inputs, labels) in enumerate(train_dataloader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            run_loss += loss.item()
            loss.backward()
            optimizer.step()

            run_processed_mb += 1

            if mb % run_mb_num == 0:
                best_accuracy = processing_run(best_accuracy, device, epoch, epochs, model, params_loc, run_id,
                                               run_loss, run_processed_mb, run_start, runs_num, val_dataloader)
                run_processed_mb = 0
                run_loss = 0
                run_id += 1
                run_start = time.time()

        epoch_end = time.time()

        if run_processed_mb != 0:
            processing_run(best_accuracy, device, epoch, epochs, model, params_loc, run_id, run_loss, run_processed_mb,
                           run_start, runs_num, val_dataloader)

        epoch_loss = round(epoch_loss / len(train_dataloader), 3)
        epoch_duration = round(epoch_end - epoch_start, 3)
        print(f"EPOCH: {epoch}/{epochs}, "
              f"AVG.M.B. LOSS: {epoch_loss}, "
              f"DURATION: {epoch_duration}s")


# CALCULATING AVG SESSION XVECTORS FOR EVERY SPEAKER IN TESTING DATASET

def get_avg_xvector(model, dataloader, device):
    avg_xvectors = torch.Tensor().to(device)

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            xvectors, _ = model(inputs)
            avg_xvectors = torch.cat((avg_xvectors, torch.mean(xvectors, 0).view(1, 512)), 0)

    return torch.mean(avg_xvectors, 0).tolist()


def get_session_xvectors(model, data_folder, chunk_size, batch_size, device):
    session_xvectors = []
    speaker_labels = []
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
            session_xvectors.append(get_avg_xvector(model, test_dataloader, device))
            speaker_labels.append(speaker_label)

        speaker_label += 1

    return session_xvectors, speaker_labels


# MAIN:
def arg_parser():
    parser = argparse.ArgumentParser(description="Script for training XVectors baseline architecture. "
                                     "Example of running script:"
                                     "python3  src/nn_models/xvectors_baseline.py -t data/vox1_dev/ -p model_params.pt")

    parser.add_argument('-t', '--train_data', required=True, help="Train data location (required)")
    parser.add_argument('-p', '--params', required=True, help="Parameters  of model location (required)")

    parser.add_argument('-v', '--val_proportion', type=float, nargs='?', default=0.15, help="Proportion of training "
                        "data, which should be used for validation (default: 0.15, min: 0.1, max: 0.5)")
    parser.add_argument('-c', '--chunks', type=int, nargs='?', default=24, help="Number of consecutive "
                        "parts of speaker recording (one part 25ms) processed by initial 1D conv layer (default: 24)")
    parser.add_argument('-e', '--epochs', type=int, nargs='?', default=10, help="Number of training epochs (default: 10)")
    parser.add_argument('-b', '--batch_size', type=int, nargs='?', default=1024, help="Mini-batch size (default: 1024)")
    parser.add_argument('-m', '--mb_in_run', type=int, nargs='?', default=100, help="Number of mini-batches in one "
                                                                                    "training run (default: 100)")
    args = parser.parse_args()
    return args


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = arg_parser()

    # training configuration:
    training_data = Path(args.train_data)
    model_params = Path(args.params)
    val_proportion = args.val_proportion
    val_proportion = 0.15 if val_proportion < 0.1 or val_proportion > 0.5 else val_proportion
    chunk_size = args.chunks
    epochs = args.epochs
    batch_size = args.batch_size
    mb_in_run = args.mb_in_run

    # model declaration:
    speakers_count = len(listdir(training_data))
    model = XVectorsBaseline(speakers_count, chunk_size).to(device)

    if model_params.is_file():
        model.load_state_dict(torch.load(model_params))
        print("Model parameters were loaded!")

    # loss and optimizer selection:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # create training and validation dataset and dataloader:
    dataset = create_training_mfcc_dataset(training_data, chunk_size)
    val_count = int(val_proportion * len(dataset))
    train_count = len(dataset) - val_count
    train_dataset, val_dataset = random_split(dataset, [train_count, val_count])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    # calling train model function:
    train_model(model, model_params, epochs, mb_in_run, train_dataloader, val_dataloader, optimizer, criterion, device)


if __name__ == '__main__':
    main()
