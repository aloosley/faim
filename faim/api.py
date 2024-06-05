from typing import List, Iterable

import numpy as np
from numpy._typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class FAIM(BaseEstimator):
    def __init__(self, thetas: List[float]) -> None:
        self._validate_thetas(thetas)
        self.thetas = np.array(thetas)

        self.is_fit = False

    @staticmethod
    def _validate_thetas(thetas: List[float]) -> None:
        if len(thetas) and not len(thetas) % 3 == 0:
            raise ValueError(
                "`thetas` must have a multiple of three values, one for each group. If only three values are provided, "
                "those same values will be applied to each group"
            )

    def fit(self, scores: Iterable[float], ground_truth: Iterable[float], group: Iterable[float]) -> None:
        ...

    def get_faim_scores(self, scores: Iterable[float], group: Iterable[float]) -> NDArray[np.float64]:
        if not self.is_fit:
            raise NotFittedError()
        ...

    def _get_barycenters(self):
        ...
