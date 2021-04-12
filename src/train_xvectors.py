import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import bob.measure as measure

from nn_models import XVectorsBaseline
from utils import DatasetBase


# DATASET CONSTRUCTION:

class TrainDataset(DatasetBase):

    def __init__(self, data_folder, feats_type, batch_size, device):
        super().__init__(data_folder, feats_type)
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        indices = np.random.permutation(np.arange(0, len(self.data)))
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
                if x.size()[1] < 15:
                    continue
                inputs.append(x.to(self.device))
                labels.append(t)
            yield inputs, labels


class EvalDataset(DatasetBase):

    def __init__(self, data_folder, feats_type, device):
        super().__init__(data_folder, feats_type)
        self.device = device

    def load(self):
        inputs1 = []
        inputs2 = []
        labels = []
        for t in range(len(self.data_mapping)):
            i1, i2 = np.random.randint(0, len(self.data_mapping[t]), size=2)
            path1, _ = self.data[self.data_mapping[t][i1]]
            path2, _ = self.data[self.data_mapping[t][i2]]
            try:
                x1 = torch.FloatTensor(np.expand_dims(np.load(path1), axis=0))
                x2 = torch.FloatTensor(np.expand_dims(np.load(path2), axis=0))
            except Exception:
                continue
            if x1.size()[1] < 15 or x2.size()[1] < 15:
                continue
            inputs1.append(x1.to(self.device))
            inputs2.append(x2.to(self.device))
            labels.append(t)
        return inputs1, inputs2, labels


# TRAINING MODEL:

def compute_accuracy(model, eval_dataset):
    model.eval()
    with torch.no_grad():
        inputs1, inputs2, labels = eval_dataset.load()
        out1 = model(inputs1)[0].cpu().detach().numpy()
        out2 = model(inputs2)[0].cpu().detach().numpy()
        scores = cosine_similarity(out1, out2)
        mask = np.eye(len(labels)).ravel()
        pos = scores.ravel()[np.nonzero(mask == 1)[0]].astype(np.float64)
        neg = scores.ravel()[np.nonzero(mask == 0)[0]].astype(np.float64)
        eer = measure.eer(neg, pos)
    model.train()
    return eer


def train_model(model, params_path, epochs, mb_in_run, train_dataset, eval_dataset, optimizer, device):

    best_eer = compute_accuracy(model, eval_dataset)
    print(f'Initial EER est.: {best_eer} %\n')
    batch_total = int(np.ceil(len(train_dataset) / train_dataset.batch_size))

    for epoch in range(1, epochs + 1, 1):
        print(f'START OF EPOCH: {epoch}/{epochs}')
        epoch_start = run_start = time.time()
        epoch_loss = 0.0
        batch_count = 0

        for inputs, labels in train_dataset:

            _, outputs = model(inputs)
            loss = F.cross_entropy(outputs, torch.tensor(labels).to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += float(loss)
            batch_count += 1

            if batch_count % mb_in_run == 0:
                eer = compute_accuracy(model, eval_dataset)
                run_duration = round(time.time() - run_start, 3)
                print(f'epoch: {epoch}/{epochs}, '
                      f'processed m.b.: {batch_count}/{batch_total}, '
                      f'avg.m.b. loss: {round(epoch_loss / batch_count, 6)}, '
                      f'EER est.: {round(100 * eer, 3)} %, '
                      f'duration: {run_duration}s')
                if eer < best_eer:
                    torch.save(model.state_dict(), params_path)
                    best_eer = eer
                run_start = time.time()

        epoch_duration = round(time.time() - epoch_start, 3)
        print(f'END OF EPOCH: {epoch}/{epochs}, '
              f'DURATION: {epoch_duration}s, '
              f'AVG.M.B. LOSS: {round(epoch_loss / batch_count, 6)}\n')


def arg_parser():
    parser = argparse.ArgumentParser(description='Script for training XVectors baseline architecture. ')
    parser.add_argument(      '--train_data', required=True, help='Train data location (required)')
    parser.add_argument(      '--eval_data', required=True, help='Evaluation data location (required)')
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
    eval_data = Path(args.eval_data)
    params_path = Path(args.params)
    epochs = args.epochs
    batch_size = args.batch_size
    mb_in_run = args.mb_in_run

    # create training dataset:
    feats_type = 'mfcc.npy'
    train_dataset = TrainDataset(training_data, feats_type, batch_size, device)
    eval_dataset = EvalDataset(eval_data, feats_type, device)

    # model declaration:
    speakers_count = len(train_dataset)
    model = XVectorsBaseline(speakers_count).to(device)

    if params_path.is_file():
        model.load_state_dict(torch.load(params_path))
        print('Model parameters were loaded!')

    # optimizer selection:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # calling train model function:
    train_model(model, params_path, epochs, mb_in_run, train_dataset, eval_dataset, optimizer, device)


if __name__ == '__main__':
    main()
