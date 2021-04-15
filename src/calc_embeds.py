import argparse
from pathlib import Path
import numpy as np
import torch
from nn_models import *
from utils import DatasetBase


def compute_embeddings(model, dataset, device):
    # TODO: REPAIR THIS FUNCTION
    for path, t in dataset.data:
        try:
            x = np.load(path)
        except Exception:
            continue
        y = model.speaker_embed(x, device=device)
        if y is not None:
            yield y.cpu().detach().numpy().ravel(), t


def main():
    parser = argparse.ArgumentParser(description='Pre-calculate speakers embeddings.')
    parser.add_argument('-i', '--input', required=True, help='Input data location (required)')
    parser.add_argument('-p', '--params', required=True, help='Parameters of embedding model location (required)')
    parser.add_argument('-o', '--output', required=True, help='Output directory for embeddings')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_class = eval(args.name)
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
