import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path


def make_scatter_plot(embeddings_path, save_plot=False, first_speaker=None, last_speaker=None):
    # loading embeddings from numpy arrays
    labels = np.load(embeddings_path / "labels.npy")
    embeddings = np.load(embeddings_path / "embeddings.npy")

    # specifying range of speakers from which will be made plot
    labels_set = set(labels)
    labels_count = len(labels_set)
    if not first_speaker:
        first_speaker = 0
    if not last_speaker:
        last_speaker = labels_count

    # subsampling (taking only embeddings from specified range of speakers)
    selected_list = sorted(labels_set)[first_speaker:last_speaker]
    selected_mask = np.in1d(labels, selected_list)
    selected_labels = labels[selected_mask]
    selected_embeddings = embeddings[selected_mask, :]

    # dimensionality reduction
    two_dim_embeddings = TSNE(n_components=2, n_jobs=-1).fit_transform(selected_embeddings)

    # making 2D scatter plot with embeddings
    colormap_stretch_value = int(100 / (last_speaker - first_speaker))
    distinctive_colors = np.full(two_dim_embeddings.shape[0], -1)
    for i, label in enumerate(selected_list):
        distinctive_colors[selected_labels == label] = i * colormap_stretch_value
    plt.scatter(two_dim_embeddings[:, 0],
                two_dim_embeddings[:, 1],
                c=distinctive_colors,
                cmap="jet")

    # saving plot into file or only showing plot
    if save_plot:
        plt.savefig(f"{embeddings_path.stem}_{first_speaker}-{last_speaker}.eps", format="eps")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for making 2D TSNE plot from speaker embeddings.")
    parser.add_argument('-i', '--input', required=True,
                        help="location of numpy arrays with embeddings and labels (required)")
    parser.add_argument('-s', '--save', action='store_true', required=False,
                        help="use this option, if you want to save plot to file (optional)")
    parser.add_argument('-f', '--first', required=False, type=int,
                        help="beginning of speakers range from which will be made plot (optional)")
    parser.add_argument('-l', '--last', required=False, type=int,
                        help="end of speakers range from which will be made plot (optional)")
    args = parser.parse_args()
    make_scatter_plot(Path(args.input),
                      args.save,
                      args.first,
                      args.last)
