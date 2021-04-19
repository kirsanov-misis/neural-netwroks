import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    targets_index = -1

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        features = np.array(df.iloc[:, :cls.targets_index])
        targets = np.array(df.iloc[:, cls.targets_index])
        return cls(features, targets)

    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 test_size: int = 0.3):
        self.features = features
        self.targets = targets
        split = train_test_split(self.features, self.targets,
                                 test_size=test_size,
                                 random_state=42)
        self.x_train = np.array(split[0])
        self.x_test = np.array(split[1])
        self.y_train = np.array(split[2])
        self.y_test = np.array(split[3])

    @property
    def x(self):
        return self.features[:, 0]

    @property
    def y(self):
        return self.features[:, 1]

    @property
    def train(self) -> 'Dataset':
        return Dataset(self.x_train, self.y_train)

    @property
    def test(self) -> 'Dataset':
        return Dataset(self.x_test, self.y_test)
