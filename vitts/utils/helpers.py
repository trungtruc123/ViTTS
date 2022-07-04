import numpy as np
import torch
from torch.nn import functional as F


class StandardScaler:
    """StandardScaler for mean-scale normalization with the given mean and scale values."""

    def __init__(self, mean: np.ndarray = None, scale: np.ndarray = None) -> None:
        self.mean_ = mean
        self.scale_ = scale

    def set_stats(self, mean, scale):
        self.mean_ = mean
        self.scale_ = scale

    def reset_stats(self):
        delattr(self, "mean_")
        delattr(self, "scale_")

    def transform(self, X):
        X = np.asarray(X)
        X -= self.mean_
        X /= self.scale_
        return X

    def inverse_transform(self, X):
        X = np.asarray(X)
        X *= self.scale_
        X += self.mean_
        return X
