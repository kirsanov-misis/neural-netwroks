from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.metrics import classification_report
from typing import Tuple
from argparse import ArgumentParser, ArgumentTypeError
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fitter import Fitter
from dataset import Dataset


def str2bool(argument):
    if isinstance(argument, bool):
        return argument
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise ArgumentTypeError('Boolean value expected.')


def init_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--enable_generated_dataset', type=str2bool,
                        default=True)
    parser.add_argument('--enable_archive_dataset', type=str2bool,
                        default=False)
    parser.add_argument('--enable_archive_squeezed_dataset', type=str2bool,
                        default=False)
    parser.add_argument('--penalty_perceptron', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=1e-4)
    parser.add_argument('--eta0', type=float, default=1.0)
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--validation_fraction', type=float, default=0.1)
    parser.add_argument('--early_stopping_perceptron', type=str2bool,
                        default=False)
    parser.add_argument('--penalty_sgd', type=str, default='l2')
    parser.add_argument('--early_stopping_sgd', type=str2bool, default=False)
    return parser


def init_perceptron_from(arguments) -> Fitter:
    return Fitter(Perceptron(random_state=42, n_jobs=-1,
                             early_stopping=arguments.early_stopping_perceptron,
                             penalty=arguments.penalty_perceptron,
                             alpha=arguments.alpha,
                             eta0=arguments.eta0,
                             tol=arguments.tol,
                             validation_fraction=arguments.validation_fraction))


def init_sgd_from(arguments) -> Fitter:
    return Fitter(SGDClassifier(random_state=42, n_jobs=-1,
                                early_stopping=arguments.early_stopping_sgd,
                                penalty=arguments.penalty_sgd))


def encode_column(dataframe: pd.DataFrame, column: int):
    dataframe.iloc[:, column] = dataframe.iloc[:, column].astype('category')
    dataframe.iloc[:, column] = dataframe.iloc[:, column].cat.codes
    return dataframe


def read_dataset(filename: str, separator: str,
                 need_to_encode: bool = False) -> Dataset:
    df = pd.read_csv(filename, sep=separator, header=None)
    if need_to_encode:
        target_column = 10
        encode_column(df, column=target_column)
    return Dataset.from_dataframe(df)


def draw_dataset(dataset: Dataset):
    plt.scatter(dataset.x, dataset.y, c=dataset.targets)


def get_array_range(array: np.ndarray) -> Tuple[float, float]:
    offset = 1
    return np.min(array) - offset, np.max(array) + offset


def get_grid(dataset: Dataset) -> np.ndarray:
    xx = dataset.x
    yy = dataset.y
    x_range = get_array_range(xx)
    y_range = get_array_range(yy)
    return np.meshgrid(np.arange(*x_range, 0.1), np.arange(*y_range, 0.1))


def reshape_grid_for_prediction(grid: np.ndarray):
    flat_x = grid[0].flatten()
    flat_y = grid[1].flatten()
    flat_x = flat_x.reshape((len(flat_x), 1))
    flat_y = flat_y.reshape((len(flat_y), 1))
    prediction_grid = np.hstack((flat_x, flat_y))
    return prediction_grid


def draw_colored_area(grid: np.ndarray, classifier: Fitter):
    prediction_grid = reshape_grid_for_prediction(grid)
    labeled_area = classifier.predict(prediction_grid)
    shape_for_draw = grid[0].shape
    plt.contourf(grid[0], grid[1], labeled_area.reshape(shape_for_draw),
                 cmap='Paired')


def draw_dataset_with_colored_area(classifier: Fitter, dataset: Dataset):
    grid = get_grid(dataset)
    draw_colored_area(grid, classifier)
    draw_dataset(dataset)
    plt.show()


def print_score(model: str, classifier: Fitter, dataset: Dataset):
    prediction = classifier.predict(dataset.features)
    print(f'Model - {model}')
    print(classification_report(dataset.targets, prediction, zero_division=0.0,
                                digits=4))


def is_n_dimensional(array: np.ndarray, n: int):
    dimensionality_index = 1
    return array.shape[dimensionality_index] == n


def perform_lab_algo(classifier: Fitter, dataset: Dataset, model_name: str):
    classifier.fit(dataset.train)
    if is_n_dimensional(dataset.features, n=2):
        draw_dataset_with_colored_area(classifier, dataset.train)
        draw_dataset_with_colored_area(classifier, dataset.test)
    print_score(model_name, classifier, dataset.test)


def process_dataset(dataset: Dataset, perceptron: Fitter,
                    sgd: Fitter):
    perform_lab_algo(perceptron, dataset, 'Perceptron')
    perform_lab_algo(sgd, dataset, 'sgd')


def main():
    arguments = init_argument_parser().parse_args()
    perceptron = init_perceptron_from(arguments)
    sgd = init_sgd_from(arguments)
    if arguments.enable_generated_dataset:
        dataset = read_dataset('generated_dataset.csv', ',')
        process_dataset(dataset, perceptron, sgd)
    if arguments.enable_archive_dataset:
        dataset = read_dataset('avila.txt', ',', need_to_encode=True)
        process_dataset(dataset, perceptron, sgd)
    if arguments.enable_archive_squeezed_dataset:
        dataset = read_dataset('avila_squeezed.txt', ',', need_to_encode=True)
        process_dataset(dataset, perceptron, sgd)


main()
