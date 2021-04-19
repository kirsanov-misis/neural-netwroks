from sklearn.linear_model import Perceptron, SGDClassifier
import numpy as np
from typing import Union
from dataset import Dataset


class Fitter:

    def __init__(self,
                 classifier: Union[Perceptron, SGDClassifier]):
        self.classifier = classifier

    def fit(self, dataset: Dataset):
        self.classifier.fit(dataset.features, dataset.targets)

    def predict(self, features: np.ndarray):
        return self.classifier.predict(features)
