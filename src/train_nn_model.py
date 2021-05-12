import math
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from nn_models import *
from utils import DatasetClasses, DatasetTriplets


def forward_with_softmax_loss(model, inputs, labels):
    _, outputs = model(inputs)
    loss = F.cross_entropy(outputs, labels)
    return loss


def euclidean_dist_without_sqrt(points1, points2):
    dist = points1 - points2
    dist = torch.pow(dist, 2)
    dist = torch.sum(dist, dim=1)
    return dist


def forward_with_angular_loss(model, anchors, positives, negatives, alpha=0.5):
    out_anchors = model(anchors)
    out_positives = model(positives)
    out_negatives = model(negatives)

    dist1 = torch.abs(euclidean_dist_without_sqrt(out_anchors, out_positives))
    xc = (out_anchors + out_positives) / 2
    dist2 = torch.abs(euclidean_dist_without_sqrt(out_negatives, xc))

    # default value for alpha is 0.5 radians (approximately 30 degrees)
    constant = 4 * (math.tan(alpha)**2)
    loss = dist1 - constant * dist2
    loss = torch.where(loss > 0, loss, torch.zeros(loss.size(0)))

    return torch.mean(loss)


def forward_with_angular_softmax_loss(model, inputs, labels):
    loss = model(inputs, labels)
    return loss


def forward_with_triplet_loss(model, anchors, positives, negatives):
    anc = model(anchors)
    pos = model(positives)
    neg = model(negatives)
    anc_norm = anc / anc.norm(dim=1, keepdim=True)
    pos_norm = pos / pos.norm(dim=1, keepdim=True)
    neg_norm = neg / neg.norm(dim=1, keepdim=True)
#    margin = 0.8
    margin = 1.4
    loss = F.triplet_margin_loss(anc_norm, pos_norm, neg_norm, margin=margin)
    return loss


def cross_validate(model, dataset, forward_with_loss_fn):
    model.eval()
    with torch.no_grad():
        loss = forward_with_loss_fn(model, *dataset.cross_val())
    model.train()
    return float(loss)


def train_model(model, params_path, epochs, mb_in_run, dataset, optimizer, forward_with_loss_fn):

    best_c_val_loss = cross_validate(model, dataset, forward_with_loss_fn)
    print(f'Initial cross-val loss: {best_c_val_loss}\n')
    batch_total = int(np.ceil(len(dataset) / dataset.batch_size))

    for epoch in range(1, epochs + 1, 1):
        print(f'START OF EPOCH: {epoch}/{epochs}')
        epoch_start = run_start = time.time()

        train_loss = 0.0
        batch_count = 0

        for batch in dataset:

            loss = forward_with_loss_fn(model, *batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += float(loss)
            batch_count += 1

            if batch_count % mb_in_run == 0:
                c_val_loss = cross_validate(model, dataset, forward_with_loss_fn)
                run_duration = round(time.time() - run_start, 3)
                print(f'epoch: {epoch}/{epochs}, '
                      f'processed m.b.: {batch_count}/{batch_total}, '
                      f'train. loss: {round(train_loss / mb_in_run, 6)}, '
                      f'cross-val. loss: {round(c_val_loss, 6)}, '
                      f'duration: {run_duration}s')
                train_loss = 0.0

                if c_val_loss < best_c_val_loss or best_c_val_loss == 0:
                    torch.save(model.state_dict(), params_path)
                    best_c_val_loss = c_val_loss
                run_start = time.time()

        epoch_duration = round(time.time() - epoch_start, 3)
        print(f'END OF EPOCH: {epoch}/{epochs}, '
              f'DURATION: {epoch_duration}s')


def arg_parser():
    parser = argparse.ArgumentParser(description='Script for training different NN models.')
    parser.add_argument('-t', '--train_data', required=True, help='Train data location (required)')
    parser.add_argument('-p', '--params', required=True, help='Parameters of model location (required)')
    parser.add_argument('-e', '--epochs', type=int, nargs='?', default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('-b', '--batch_size', type=int, nargs='?', default=64, help='Mini-batch size (default: 64)')
    parser.add_argument('-m', '--mb_in_run', type=int, nargs='?', default=50, help='Number of mini-batches in one '
                                                                                   'training run (default: 50)')
    parser.add_argument('--loss', required=True, help='Loss used for training [softmax | a_softmax | triplet | angular]')
    parser.add_argument('--lr', type=float, nargs='?', default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--optim', required=True, nargs='?', default='SGD', help='Optimizer [SGD | Adam]')
    args = parser.parse_args()
    return args


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = arg_parser()

    # training configuration:
    train_data = Path(args.train_data)
    params_path = Path(args.params)
    epochs = args.epochs
    batch_size = args.batch_size
    mb_in_run = args.mb_in_run
    lr = args.lr
    loss_type = args.loss

    # set respective dataset, model and loss type:
    if loss_type == 'triplet' or loss_type == 'angular':
        dataset_class = DatasetTriplets
        model_class = XVectors

        if loss_type == 'triplet':
            forward_with_loss_fn = forward_with_triplet_loss
        else:
            forward_with_loss_fn = forward_with_angular_loss

    elif loss_type == 'softmax' or loss_type == 'a_softmax':
        dataset_class = DatasetClasses

        if loss_type == 'softmax':
            model_class = XVectorsSoftmax
            forward_with_loss_fn = forward_with_softmax_loss
        else:
            model_class = XVectorsAngularSoftmax
            forward_with_loss_fn = forward_with_angular_softmax_loss
    else:
        print('Invalid loss name', file=sys.stderr)
        exit(1)

    feats_type = 'mfcc'
    min_sample_length = model_class.min_sample_length()
    train_dataset = dataset_class(train_data, feats_type, batch_size, device, min_sample_length)

    # model initialization:
    model = model_class().to(device)
    if params_path.is_file():
        model.load_state_dict(torch.load(params_path))
        print('Model parameters were loaded!')

    # optimizer selection:
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        print('Invalid optimizer name', file=sys.stderr)
        exit(1)

    # calling train model function:
    train_model(model, params_path, epochs, mb_in_run, train_dataset, optimizer, forward_with_loss_fn)


if __name__ == '__main__':
    main()
