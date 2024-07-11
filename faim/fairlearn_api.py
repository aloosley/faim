# ToDo: (WIP) Create an API that conforms to ML community standards
from decimal import Decimal
from functools import lru_cache
from typing import Iterable, Protocol, Any, cast, Optional
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
    """
    Parameters
    ----------
    thetas
        Either a dictionary with group ids and theta values for each group, or a list of three theta values
        to be applied to each group. Thetas determine how far a score distribution is to be moved towards
            the barycenter representing the three mutually exclusive fairness criteria:
                1. Calibration between groups (scores actually correspond to probability of positive)
                2. Balance for the negative class (average score of truly negative individuals equal across groups)
                3. Balance for the positive class (average score of truly positive individuals equal across groups)
    score_discretization_step
        Defines how scores are discretized for optimal transport, i.e. a value of 0.01 leads to a score discretization
        of 0.0, 0.01, 0.02, ... 0.99, 1.00 (note that the algorithm only accepts normalized scores)
    optimal_transport_regularization
        Regularization parameter for optimal transport, see POT documentation (`reg` argument) for details
        (https://pythonot.github.io/all.html)
    random_generator
        Numpy random generator or integer seed - provide when exact reproducibility is desired

    Examples
    --------


    """

    def __init__(
        self,
        thetas: list[float] | dict[Any, list[float]],
        score_discretization_step: float = 0.01,
        score_map: dict[Any, NDArray[np.float64]] | None = None,
        optimal_transport_regularization: float = 0.01,
        random_generator: Generator | int | None = None,
    ) -> None:
        # Validate and set FAIM parameters
        self._validate_thetas(thetas)

        if random_generator is None:
            random_generator = Generator(PCG64(43))
        elif isinstance(random_generator, int):
            random_generator = Generator(PCG64(random_generator))

        self.thetas: NDArray[np.float64] = np.array(thetas)
        self.score_discretization_step = score_discretization_step
        self.optimal_transport_regularization = optimal_transport_regularization

        # Use score map if provided
        if score_map is None:
            self.prefit = False
        else:
            self.prefit = True
        self.score_map = score_map

        # Implementation parameters
        self.random_generator = random_generator

    def fit(
        self, y_scores: Iterable[float], y_ground_truth: Iterable[Any], *, sensitive_features: Iterable[Any]
    ) -> "FAIM":
        y_scores, y_ground_truth, sensitive_features = self._validate_and_format_inputs(
            y_scores, y_ground_truth, sensitive_features, score_discretization_step=self.score_discretization_step
        )
        y_scores_discretized = self._discretize_scores(y_scores)

        # Compute mu_A (calibrated score distribution)
        mu_a = self._compute_mu_a(y_scores_discretized, y_ground_truth, sensitive_features)

        return self

    def predict(self, y_scores: Iterable[Any], *, sensitive_features: Iterable[Any]) -> NDArray[np.float64]:
        discrete_y_scores = y_scores[np.digitize(y_scores, self.score_discretization) - 1]

    @staticmethod
    def _validate_thetas(thetas: list[float] | dict[Any, list[float]]) -> None:
        if isinstance(thetas, list) and len(thetas) != 3:
            raise ValueError(
                "`thetas` must have three values if provided as a list. The three values will be applied to each group."
            )
        if isinstance(thetas, dict):
            for group_id, group_thetas in thetas.items():
                if len(group_thetas) != 3:
                    raise ValueError(
                        f"There must be 3 theta values for each group  but group {group_id} has "
                        f"{len(group_thetas)} theta values."
                    )

    def get_faim_scores(self, scores: Iterable[float], group: Iterable[float]) -> NDArray[np.float64]:
        if not self.is_fit:
            raise NotFittedError()
        ...

    @property
    @lru_cache
    def normalized_discrete_score_values(self) -> NDArray[np.float64]:
        return self._get_normalized_discrete_score_values(score_discretization_step=self.score_discretization_step)

    @staticmethod
    def _get_normalized_discrete_score_values(score_discretization_step: float) -> NDArray[np.float64]:
        return np.round(
            np.arange(0, 1 + score_discretization_step, score_discretization_step),
            decimals=int(np.ceil(-np.log10(score_discretization_step)) + 1),
        )

    def _discretize_scores(self, scores: NDArray[np.float64], step: Optional[float] = None) -> NDArray[np.float64]:
        normalized_discrete_score_values: NDArray[np.float64]
        if step is None:
            normalized_discrete_score_values = self.normalized_discrete_score_values
        else:
            normalized_discrete_score_values = self._get_normalized_discrete_score_values(step)

        return self.normalized_discrete_score_values[np.digitize(scores, normalized_discrete_score_values) - 1]

    def _compute_mu_a(
        self,
        discrete_y_scores: NDArray[np.float64],
        y_ground_truth: NDArray[np.float64],
        sensitive_features: NDArray[np.float64],
    ) -> pd.DataFrame:
        """Compute calibrated score distribution by group."""
        # Get calibrated scores (known as S_A in paper)
        calibrated_scores = FAIM._compute_calibrated_scores(discrete_y_scores, y_ground_truth, sensitive_features)

        # Return distributions (index = discrete scores, columns = sensitive groups)
        return pd.DataFrame(
            data={
                sensitive_feature: np.histogram(
                    calibrated_scores[sensitive_features == sensitive_feature],
                    bins=self.normalized_discrete_score_values,
                    density=False,
                )[0]
                for sensitive_feature in np.unique(sensitive_features)
            },
            index=self.normalized_discrete_score_values[:-1],
        )

    @staticmethod
    def _compute_calibrated_scores(
        discrete_y_scores: NDArray[np.float64],
        y_ground_truth: NDArray[np.float64],
        sensitive_features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        df = pd.DataFrame(
            {
                "discrete_y_scores": discrete_y_scores,
                "y_ground_truth": y_ground_truth,
                "sensitive_features": sensitive_features,
            }
        )
        calibrated_score_by_sensitive_feature_and_discrete_score = df.groupby(
            ["sensitive_features", "discrete_y_scores"], sort=False
        ).agg(calibrated_scores=("y_ground_truth", "mean"))
        return df.merge(
            calibrated_score_by_sensitive_feature_and_discrete_score,
            how="left",
            left_on=["sensitive_features", "discrete_y_scores"],
            right_index=True,
        )["calibrated_scores"].to_numpy()

    def _compute_sigma_b_and_c_score_distributions(
        self,
        discrete_y_scores: NDArray[np.float64],
        y_ground_truth: NDArray[np.float64],
        sensitive_features: NDArray[np.float64],
    ):
        data = pd.DataFrame(
            {
                "discrete_y_scores": discrete_y_scores,
                "y_ground_truth": y_ground_truth,
                "sensitive_features": sensitive_features,
            }
        )

        # Get discretized score counts by sensitive group and ground truth value
        sensitive_groups = np.unique(sensitive_features)
        ground_truth_values = np.unique(y_ground_truth)
        cartesion_product_of_index_features_with_discrete_score_basis = pd.MultiIndex.from_product(
            [sensitive_groups, ground_truth_values, self.normalized_discrete_score_values]
        )
        grouped_discretized_score_counts = (
            data.groupby(["sensitive_features", "y_ground_truth", "discrete_y_scores"], sort=True)
            .agg(score_count=("discrete_y_scores", "count"))
            .reindex(cartesion_product_of_index_features_with_discrete_score_basis)
            .fillna(0)
        )

        # Normalized score counts
        for group in sensitive_groups:
            for ground_truth_value in ground_truth_values:
                grouped_discretized_score_counts.loc[group, ground_truth_value] = (
                    grouped_discretized_score_counts.loc[group, ground_truth_value]
                    / grouped_discretized_score_counts.loc[group, ground_truth_value].sum()
                ).to_numpy()

        grouped_discretized_score_counts.index.names = ["sensitive_group", "ground_truth", "y_score"]
        grouped_discretized_score_counts.columns = ["fraction"]
        return grouped_discretized_score_counts

    def _get_barycenters(self):
        ...

    @staticmethod
    def histogram_by_sensitive_feature(y_scores, sensitive_features) -> pd.DataFrame:
        return pd.DataFrame(
            {
                sensitive_feature: y_scores[sensitive_features == sensitive_feature]
                for sensitive_feature in np.unique(sensitive_features)
            }
        )

    def _validate_and_format_inputs(
        self,
        y_scores: Iterable[float],
        y_ground_truth: Optional[Iterable[Any]],
        sensitive_features: Iterable[Any],
        score_discretization_step: float,
    ) -> tuple[NDArray[np.float64], Optional[NDArray[np.float64]], NDArray[np.float64]]:
        y_scores = np.array(y_scores)
        if y_ground_truth is not None:
            y_ground_truth = np.array(y_ground_truth, dtype=bool)
        sensitive_features = np.array(sensitive_features)

        if not 0 <= score_discretization_step <= 0.2:
            raise ValueError(
                f"score_discretization_step must be between 0 and 0.1 ({score_discretization_step} passed), ."
            )

        if y_scores.max() > 1 or y_scores.min() < 0:
            raise ValueError(f"y_scores must be between 0 and 1 (min={y_scores.min()}, max={y_scores.max()}).")

        if len(y_scores) != len(sensitive_features):
            raise ValueError(
                f"Length of y_scores ({len(y_scores)}) and sensitive_features ({len(sensitive_features)}) must be equal."
            )

        if y_ground_truth is not None and len(y_ground_truth) != len(y_scores):
            raise ValueError(
                f"Length of y_ground_truth ({len(y_ground_truth)}) and y_scores ({len(y_scores)}) must be equal."
            )

        return y_scores, y_ground_truth, sensitive_features


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
            self.faim.fit(y_scores=y_scores, y_ground_truth=y, sensitive_features=sensitive_features)

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
