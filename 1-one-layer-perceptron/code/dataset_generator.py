from sklearn.datasets import make_blobs
from argparse import ArgumentParser
from numpy import ndarray, hstack, savetxt
from typing import Tuple


def init_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='If int, it is the total number of points equally '
                             'divided among clusters. If array-like, each '
                             'element of the sequence indicates the number of '
                             'samples per cluster.')
    parser.add_argument('--n_features', type=int, default=2,
                        help='The number of features for each sample.')
    parser.add_argument('--classes', type=int, default=5,
                        help='The number of classes (or labels) of the '
                             'classification problem.')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Shuffle the samples')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Determines random number generation for dataset '
                             'creation. Pass an int for reproducible output '
                             'across multiple function calls')
    return parser


def generate_blobs(n_samples: int, n_features: int, n_classes: int,
                   shuffle: bool, random_state: int) -> Tuple[ndarray,
                                                              ndarray]:
    return make_blobs(n_samples=n_samples, centers=n_classes,
                      n_features=n_features,
                      shuffle=shuffle, random_state=random_state)


def main():
    parser = init_argument_parser()
    arguments = parser.parse_args()
    features, targets = generate_blobs(arguments.n_samples,
                                       arguments.n_features,
                                       arguments.classes,
                                       arguments.shuffle,
                                       arguments.random_state)
    targets = targets.reshape((targets.shape[0], 1))
    savetxt('generated_dataset.csv', hstack([features, targets]), delimiter=',',
            fmt=['%lf', '%lf', '%d'], )


if __name__ == '__main__':
    main()
