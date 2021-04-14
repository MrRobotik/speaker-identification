import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from nn_models import XVectorsBaseline
from utils import DatasetBase


# DATASET CONSTRUCTION:

class TrainDataset(DatasetBase):

    def __init__(self, data_folder, feats_type, batch_size, device):
        super().__init__(data_folder, feats_type)
        self.batch_size = batch_size
        self.device = device
        self.cross_val_indices = []
        for utters in self.speaker_data:
            i = utters[0]
            self.cross_val_indices.append(i)
        cvis = set(self.cross_val_indices)
        self.train_indices = [i for i in range(len(self.data)) if i not in cvis]
        self.cross_val_indices = np.asarray(self.cross_val_indices)
        self.train_indices = np.asarray(self.train_indices)

    def __iter__(self):
        indices = np.random.permutation(self.train_indices)
        for batch_start in range(0, len(indices), self.batch_size):
            batch_end = batch_start + self.batch_size
            inputs = []
            labels = []
            for i in indices[batch_start:batch_end]:
                path, t = self.data[i]
                try:
                    x = torch.FloatTensor(np.expand_dims(np.load(path), axis=0))
                except Exception:
                    continue
                if x.size()[2] < 15:
                    continue
                inputs.append(x.to(self.device))
                labels.append(t)
            yield inputs, labels

    def cross_val(self):
        labels = []
        inputs = []
        for i in self.cross_val_indices:
            path, t = self.data[i]
            try:
                x = torch.FloatTensor(np.expand_dims(np.load(path), axis=0))
            except Exception:
                continue
            if x.size()[2] < 15:
                continue
            inputs.append(x.to(self.device))
            labels.append(t)
        return inputs, labels


# TRAINING MODEL:

def cross_validate(model, dataset):
    model.eval()
    with torch.no_grad():
        inputs, labels = dataset.cross_val()
        _, outputs = model(inputs)
        loss = F.cross_entropy(outputs.cpu(), torch.tensor(labels))
    model.train()
    return float(loss)


def train_model(model, params_path, epochs, mb_in_run, dataset, optimizer, device):

    best_c_val_loss = cross_validate(model, dataset)
    print(f'Initial cross-val loss: {best_c_val_loss}\n')
    batch_total = int(np.ceil(len(dataset) / dataset.batch_size))

    for epoch in range(1, epochs + 1, 1):
        print(f'START OF EPOCH: {epoch}/{epochs}')
        epoch_start = run_start = time.time()

        train_loss = 0.0
        batch_count = 0

        for inputs, labels in dataset:

            _, outputs = model(inputs)
            loss = F.cross_entropy(outputs, torch.tensor(labels).to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += float(loss)
            batch_count += 1

            if batch_count % mb_in_run == 0:
                c_val_loss = cross_validate(model, dataset)
                run_duration = round(time.time() - run_start, 3)
                print(f'epoch: {epoch}/{epochs}, '
                      f'processed m.b.: {batch_count}/{batch_total}, '
                      f'train. loss: {round(train_loss / mb_in_run, 6)}, '
                      f'cross-val. loss: {round(c_val_loss, 6)}, '
                      f'duration: {run_duration}s')
                train_loss = 0.0

                if c_val_loss < best_c_val_loss:
                    torch.save(model.state_dict(), params_path)
                    best_c_val_loss = c_val_loss
                run_start = time.time()

        epoch_duration = round(time.time() - epoch_start, 3)
        print(f'END OF EPOCH: {epoch}/{epochs}, '
              f'DURATION: {epoch_duration}s')


def arg_parser():
    parser = argparse.ArgumentParser(description='Script for training XVectors baseline architecture.')
    parser.add_argument('-t', '--train_data', required=True, help='Train data location (required)')
    parser.add_argument('-p', '--params', required=True, help='Parameters of model location (required)')
    parser.add_argument('-e', '--epochs', type=int, nargs='?', default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('-b', '--batch_size', type=int, nargs='?', default=64, help='Mini-batch size (default: 64)')
    parser.add_argument('-m', '--mb_in_run', type=int, nargs='?', default=50, help='Number of mini-batches in one '
                                                                                    'training run (default: 50)')
    args = parser.parse_args()
    return args


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = arg_parser()

    # training configuration:
    training_data = Path(args.train_data)
    params_path = Path(args.params)
    epochs = args.epochs
    batch_size = args.batch_size
    mb_in_run = args.mb_in_run

    # create training dataset:
    feats_type = 'mfcc'
    train_dataset = TrainDataset(training_data, feats_type, batch_size, device)

    # model declaration:
    speakers_count = train_dataset.speakers_count()
    model = XVectorsBaseline(speakers_count).to(device)

    if params_path.is_file():
        model.load_state_dict(torch.load(params_path))
        print('Model parameters were loaded!')

    # optimizer selection:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # calling train model function:
    train_model(model, params_path, epochs, mb_in_run, train_dataset, optimizer, device)


if __name__ == '__main__':
    main()
