import argparse
from pathlib import Path
import numpy as np
import torch
from nn_models import *
from utils import DatasetBase


def compute_embeddings(model, dataset, device):
    for path, t in dataset.data:
        try:
            x = torch.FloatTensor(np.expand_dims(np.load(path), axis=0))
        except Exception:
            continue
        if x.size()[2] < model.min_sample_length():
            continue
        y = model([x.to(device)])
        if y is not None:
            if device == 'cuda':
                y = y.cpu()
            yield y.detach().numpy().ravel(), t


def main():
    parser = argparse.ArgumentParser(description='Pre-calculate speakers embeddings.')
    parser.add_argument('-i', '--input', required=True, help='Input data location (required)')
    parser.add_argument('-p', '--params', required=True, help='Parameters of embedding model location (required)')
    parser.add_argument('-o', '--output', required=True, help='Output directory for embeddings')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feats_type = 'mfcc'
    dataset = DatasetBase(Path(args.input), feats_type)
    model = XVectors().to(device).eval()
    model.load_state_dict(torch.load(args.params), strict=False)
    output_dir = Path(args.output)

    embeddings = []
    labels = []

    counter = 0
    for x, t in compute_embeddings(model, dataset, device):
        embeddings.append(x)
        labels.append(t)
        counter += 1
        progress = 100 * (counter / len(dataset))
        print('progress: %.2f %%' % progress, end='\r')

    print('progress: 100 %')
    embeddings = np.vstack(embeddings)
    labels = np.asarray(labels)
    np.save(output_dir / 'embeddings.npy', embeddings)
    np.save(output_dir / 'labels.npy', labels)


if __name__ == '__main__':
    main()
