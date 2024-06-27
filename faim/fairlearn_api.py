# ToDo: (WIP) Create an API that conforms to ML community standards

from typing import Iterable, Protocol, Any, cast
from warnings import warn

import numpy as np
import pandas as pd
from fairlearn.postprocessing._constants import BASE_ESTIMATOR_NONE_ERROR_MESSAGE, BASE_ESTIMATOR_NOT_FITTED_WARNING
from fairlearn.utils._common import _get_soft_predictions
from fairlearn.utils._input_validation import _validate_and_reformat_input
from numpy._typing import NDArray
from numpy.random import Generator, PCG64
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None):
        ...

    def predict(self, X):
        ...

    def score(self, X, y, sample_weight=None):
        ...

    def set_params(self, **params):
        ...


class FAIM:
    def __init__(
        self, thetas: list[float], prefit: bool = False, random_generator: Generator | int | None = None
    ) -> None:
        self._validate_thetas(thetas)
        if random_generator is None:
            random_generator = Generator(PCG64(43))
        elif isinstance(random_generator, int):
            random_generator = Generator(PCG64(random_generator))

        self.thetas: NDArray[np.float64] = np.array(thetas)
        self.prefit = prefit

        self.random_generator = random_generator

    def fit(
        self, y_scores: Iterable[Any], y_groundtruth: Iterable[Any], *, sensitive_features: Iterable[Any]
    ) -> "FAIM":
        ...

    def predict(self, y_scores: Iterable[Any], *, sensitive_features: Iterable[Any]) -> NDArray[np.float64]:
        ...

    @staticmethod
    def _validate_thetas(thetas: list[float]) -> None:
        if len(thetas) and not len(thetas) % 3 == 0:
            raise ValueError(
                "`thetas` must have a multiple of three values, one for each group. If only three values are provided, "
                "those same values will be applied to each group"
            )

    def get_faim_scores(self, scores: Iterable[float], group: Iterable[float]) -> NDArray[np.float64]:
        if not self.is_fit:
            raise NotFittedError()
        ...

    def _get_barycenters(self):
        ...


class FaimEstimator(BaseEstimator):
    """A classifier based on the FAir Interpolation Method (FAIM).

    The classifier is trained by using optimal transport to determine
    a score map based for each protected group based on the choice
    of wishing to satisfy a certain balance three fairness criteria:
    A. Calibration between groups (scores actually correspond to probability of positive)
    B. Balance for the negative class (average score of truly negative individuals equal across groups)
    C. Balance for the positive class (average score of truly positive individuals equal across groups)

    See our paper for more details: <https://arxiv.org/pdf/2212.00469>

    Parameters
    ----------
    estimator : object
        A `scikit-learn compatible estimator
        <https://scikit-learn.org/stable/developers/develop.html#estimators>`_
        whose output is postprocessed.

    thetas : array
        Desired balance between fairness criteria A, B, C (see above).

        For example, [0.5, 0.25, 0.25] would mean desiring all three fairness
        criteria, but optimizing for calibrated scores twice as much as optimizing for balance
        for the negative and positive class.

    Examples
    --------
    >>> from fairlearn.postprocessing import ThresholdOptimizer
    >>> from sklearn.linear_model import LogisticRegression
    >>> X                  = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    >>> y_scores                  = [ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  1 ,  0 ,  0 ,  0 ]
    >>> sensitive_features = ["a", "b", "a", "a", "b", "a", "b", "b", "a", "b"]
    >>> unmitigated_lr = LogisticRegression().fit(X, y_scores)
    >>> faim = FAIM(thetas = [0.5, 0.25, 0.25])
    >>> faim_lr = FaimEstimator(
    ...     estimator=unmitigated_lr,
    ...     faim=faim
    ... )
    >>> faim_lr.fit_predict(X, y_scores, sensitive_features=sensitive_features)
    """

    def __init__(self, estimator: ScikitModel, faim: FAIM, *, prefit: bool = False) -> None:

        self.estimator = estimator
        self.faim = faim

        self.prefit = prefit

        self.is_fit = False

    def fit(self, X, y, *, sensitive_features, **kwargs: dict[Any, Any]) -> "FaimEstimator":
        """Fit the model.

        The fit is based on training features and labels, sensitive features,
        as well as the fairness-unaware predictor or estimator. If an estimator was passed
        in the constructor this fit method will call `fit(X, y_scores, **kwargs)` on said estimator.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            The feature matrix
        y : numpy.ndarray, pandas.DataFrame, pandas.Series, or list
            The label vector
        sensitive_features : numpy.ndarray, list, pandas.DataFrame, or pandas.Series
            sensitive features to identify groups by
        """
        if self.estimator is None:
            raise ValueError(BASE_ESTIMATOR_NONE_ERROR_MESSAGE)

        _, _, sensitive_feature_vector, _ = _validate_and_reformat_input(
            X,
            y,
            sensitive_features=sensitive_features,
            enforce_binary_labels=True,
        )

        # postprocessing can't handle 0/1 as floating point numbers, so this
        # converts it to int
        if type(y) in [np.ndarray, pd.DataFrame, pd.Series]:
            y = y.astype(int)
        else:
            y = [int(y_val) for y_val in y]

        self.estimator_: ScikitModel
        if not self.prefit:
            # Following is on two lines due to issue when estimator comes from
            # TensorFlow
            self.estimator_ = cast(ScikitModel, clone(self.estimator))
            self.estimator_.fit(X, y, **kwargs)
        else:
            try:
                check_is_fitted(self.estimator)
            except NotFittedError:
                warn(BASE_ESTIMATOR_NOT_FITTED_WARNING.format(type(self).__name__))
            self.estimator_ = self.estimator

        y_scores = _get_soft_predictions(estimator=self.estimator_, X=X, predict_method="predict")

        if not self.faim.prefit:
            self.faim.fit(y_scores=y_scores, y_groundtruth=y, sensitive_features=sensitive_features)

        return self

    def predict(self, X, *, sensitive_features, random_state=None):
        """Predict label for each sample in X while taking into account \
            sensitive features.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            feature matrix
        sensitive_features : numpy.ndarray, list, pandas.DataFrame, pandas.Series
            sensitive features to identify groups by
        random_state : int or :class:`numpy.random.RandomState` instance, default=None
            Controls random numbers used for randomized predictions. Pass an
            int for reproducible output across multiple function calls.

        Returns
        -------
        numpy.ndarray
            The prediction in the form of a scalar or vector.
            If `X` represents the data for a single example the result will be
            a scalar. Otherwise the result will be a vector.
        """
        check_is_fitted(self)
        return self.faim.predict(
            y_scores=_get_soft_predictions(estimator=self.estimator_, X=X, predict_method="predict"),
            sensitive_features=sensitive_features,
        )

    def fit_predict(self, X, y, sensitive_features) -> NDArray[np.float64]:
        return self.fit(X, y, sensitive_features=sensitive_features).predict(X, sensitive_features=sensitive_features)
