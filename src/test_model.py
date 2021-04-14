import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import bob.measure as measure


def evaluate_eer(inputs, labels):
    scores_mat = cosine_similarity(inputs, inputs)
    negatives = []
    positives = []

    for i in range(scores_mat.shape[0]):
        labels_others = np.delete(labels, i)
        pos_indices = np.nonzero(labels_others == labels[i])[0]
        neg_indices = np.nonzero(labels_others != labels[i])[0]
        positives.append(scores_mat[i, pos_indices].astype(np.float64))
        negatives.append(scores_mat[i, neg_indices].astype(np.float64))

    positives = np.concatenate(positives)
    negatives = np.concatenate(negatives)
    return measure.eer(negatives, positives)


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on pre-calculated embeddings.')
    parser.add_argument('-i', '--input', required=True, help='Pre-calc. embeddings directory (required)')
    args = parser.parse_args()
    input_dir = Path(args.input)
    embeddings = np.load(input_dir / 'embeddings.npy')
    labels = np.load(input_dir / 'labels.npy')
    eer = evaluate_eer(embeddings, labels)
    print('EER (cosine sim.): {}'.format(eer))


if __name__ == '__main__':
    main()
