# Modified from: https://github.com/sbi-benchmark/sbibm/blob/55edfab6e34cdfe3eba0022cc1382ba5e3e3d0dd/sbibm/metrics/c2st.py

import numpy as np
import torch
from numpy.random import RandomState
from numpy.typing import NDArray
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier


def c2st(
    X: NDArray,
    Y: NDArray,
    /,
    *,
    random_state: RandomState,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: float | None = None,
) -> NDArray:
    """Classifier-based 2-sample test returning accuracy

    Trains classifiers with N-fold cross-validation [1]. Scikit learn MLPClassifier are
    used, with 2 hidden layers of 10x dim each, where dim is the dimensionality of the
    samples X and Y.

    Args:
        X: Sample 1
        Y: Sample 2
        random_state: Random number generator
        n_folds: Number of folds
        z_score: Z-scoring using X
        noise_scale: If passed, will add Gaussian noise with std noise_scale to samples

    References:
        [1]: https://scikit-learn.org/stable/modules/cross_validation.html
    """
    if z_score:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * random_state.standard_normal(X.shape)
        Y += noise_scale * random_state.standard_normal(Y.shape)

    ndim = X.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=10000,
        solver="adam",
        random_state=random_state,
    )

    data = np.concatenate((X, Y))
    target = np.concatenate(
        (
            np.zeros((X.shape[0],)),
            np.ones((Y.shape[0],)),
        )
    )

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    return np.mean(scores)


def c2st_auc(
    X: NDArray,
    Y: NDArray,
    /,
    *,
    random_state: RandomState,
    n_folds: int = 5,
    z_score: bool = True,
    noise_scale: float | None = None,
) -> NDArray:
    """Classifier-based 2-sample test returning AUC (area under curve)

    Same as c2st, except that it returns ROC AUC rather than accuracy

    Args:
        X: Sample 1
        Y: Sample 2
        random_state: Random number generator
        n_folds: Number of folds
        z_score: Z-scoring using X
        noise_scale: If passed, will add Gaussian noise with std noise_scale to samples

    Returns:
        Metric
    """
    return c2st(
        X,
        Y,
        random_state=random_state,
        n_folds=n_folds,
        scoring="roc_auc",
        z_score=z_score,
        noise_scale=noise_scale,
    )
