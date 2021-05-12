import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import det_curve, DetCurveDisplay
import bob.measure as measure
import matplotlib.pyplot as plt


def prepare_pos_neg(labels, scores_mat):
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
    return positives, negatives


def evaluate_eer(negatives, positives):
    return measure.eer(negatives, positives)


def show_det_curve(negatives, positives):
    y = [-1 for i in range(len(negatives))] + \
        [+1 for i in range(len(positives))]
    scores = np.concatenate((negatives, positives))
    fpr, fnr, _ = det_curve(y, scores)
    display = DetCurveDisplay(fpr=fpr, fnr=fnr)
    display.plot()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on pre-calculated embeddings.')
    parser.add_argument('-i', '--input', required=True, help='Pre-calc. embeddings directory (required)')
    args = parser.parse_args()
    input_dir = Path(args.input)
    embeddings = np.load(input_dir / 'embeddings.npy')
    labels = np.load(input_dir / 'labels.npy')
    scores_mat = cosine_similarity(embeddings, embeddings)
    positives, negatives = prepare_pos_neg(labels, scores_mat)
    eer = evaluate_eer(negatives, positives)
    print('EER (cosine sim.): {}'.format(eer))
    show_det_curve(negatives, positives)


if __name__ == '__main__':
    main()
