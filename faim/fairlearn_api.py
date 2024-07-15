# ToDo: (WIP) Create an API that conforms to ML community standards
from copy import deepcopy
from functools import lru_cache
from typing import Iterable, Protocol, Any, cast, Optional
from warnings import warn

import numpy as np
import ot
import pandas as pd
from fairlearn.postprocessing._constants import BASE_ESTIMATOR_NONE_ERROR_MESSAGE, BASE_ESTIMATOR_NOT_FITTED_WARNING
from fairlearn.utils._common import _get_soft_predictions
from fairlearn.utils._input_validation import _validate_and_reformat_input
from numpy._typing import NDArray, ArrayLike
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
    discrete_fair_scores_by_group
        This is normally determined by fitting the data, however, you can also pass scores for each group directly
        to skip the fitting step and immediate use the model for score mapping. Each value of these scores corresponds
        to the score that each value of self.normalized_discrete_score_values should map to.
    optimal_transport_regularization
        Regularization parameter for optimal transport, see POT documentation (`reg` argument) for details
        (https://pythonot.github.io/all.html)
    random_generator
        Numpy random generator or integer seed - provide when exact reproducibility is desired

    Examples
    --------
    >>> from faim.fairlearn_api import FAIM
    >>> faim = FAIM(thetas=[0.5, 0.25, 0.25], score_discretization_step=0.01)
    >>> faim.fit(y_scores, y_ground_truth, sensitive_features)
    >>> fair_y_scores = faim.predict(y_scores, sensitive_features)

    """

    def __init__(
        self,
        thetas: list[float] | dict[Any, list[float]],
        score_discretization_step: float = 0.01,
        discrete_fair_scores_by_group: NDArray[np.float64] | None = None,
        optimal_transport_regularization: float = 0.001,
        random_generator: Generator | int | None = None,
    ) -> None:
        # Validate and set FAIM parameters
        self._validate_thetas(thetas)

        if random_generator is None:
            random_generator = Generator(PCG64(43))
        elif isinstance(random_generator, int):
            random_generator = Generator(PCG64(random_generator))

        self.thetas = thetas
        self.score_discretization_step = score_discretization_step
        self.optimal_transport_regularization = optimal_transport_regularization

        # Use score map if provided
        if discrete_fair_scores_by_group is None:
            self.prefit = False
        else:
            self.prefit = True
        self.discrete_fair_scores_by_group = discrete_fair_scores_by_group

        # Implementation parameters
        self.random_generator = random_generator

    def fit(
        self, y_scores: Iterable[float], y_ground_truth: Iterable[Any], *, sensitive_features: Iterable[Any]
    ) -> "FAIM":
        y_scores, y_ground_truth, sensitive_features = self._validate_and_format_inputs(
            y_scores, y_ground_truth, sensitive_features, score_discretization_step=self.score_discretization_step
        )
        discrete_y_scores, discrete_y_score_indices = self._get_discrete_scores_and_indices(y_scores)

        # Compute mu_t^A, mu_t^b, and mu_t^c (target score distributions for each fairness criterion)
        mu_a_per_group = self._compute_mu_a(discrete_y_scores, y_ground_truth, sensitive_features)
        mu_b_per_group, mu_c_per_group = self._compute_mu_b_and_c(
            discrete_y_scores=discrete_y_scores,
            discrete_y_score_indices=discrete_y_score_indices,
            y_ground_truth=y_ground_truth,
            sensitive_features=sensitive_features,
        )

        # Map thetas to dict keyed by group
        sensitive_groups = np.unique(sensitive_features)
        thetas = deepcopy(self.thetas)
        if not isinstance(thetas, dict):
            thetas = {group: thetas for group in sensitive_groups}

        self.discrete_fair_scores_by_group: dict[Any, NDArray[np.float64]] = {}
        for sensitive_group in sensitive_groups:
            group_mus = np.stack(
                (mu_a_per_group[sensitive_group], mu_b_per_group[sensitive_group], mu_c_per_group[sensitive_group]),
                axis=1,
            )
            faim_barycenter = ot.bregman.barycenter(
                A=np.divide(group_mus, group_mus.sum(axis=0)),
                M=self._ot_loss_matrix,
                reg=self.optimal_transport_regularization,
                weights=thetas[sensitive_group],
            )

            self.discrete_fair_scores_by_group[sensitive_group] = self._fill_discrete_fair_score_map(
                self._get_discrete_fair_score_map(
                    scores=y_scores,
                    fair_score_distribution=faim_barycenter,
                )
            )

    def predict(self, y_scores: Iterable[Any], *, sensitive_features: Iterable[Any]) -> NDArray[np.float64]:
        y_scores, y_ground_truth, sensitive_features = self._validate_and_format_inputs(
            y_scores, None, sensitive_features, score_discretization_step=self.score_discretization_step
        )
        discretized_y_scores, discretized_y_score_indices = self._get_discrete_scores_and_indices(y_scores)

        # Map scores to fair scores
        discretized_fair_y_scores = deepcopy(discretized_y_scores)
        for sensitive_group in np.unique(sensitive_features):
            group_score_mask = sensitive_features == sensitive_group
            discretized_fair_y_scores[group_score_mask] = self.discrete_fair_scores_by_group[sensitive_group][
                discretized_y_score_indices[group_score_mask]
            ]

        return discretized_fair_y_scores

    @property
    @lru_cache
    def discrete_score_bins(self) -> NDArray[np.float64]:
        return self._get_discrete_score_bins(score_discretization_step=self.score_discretization_step)

    @property
    @lru_cache
    def _ot_loss_matrix(self) -> NDArray[np.float64]:
        loss_matrix = ot.utils.dist0(len(self.discrete_score_bins))
        return loss_matrix / loss_matrix.max()

    def _get_discrete_scores_and_indices(
        self, scores: NDArray[np.float64], step: Optional[float] = None
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Discretize scores based on discrete_score_bins."""
        discrete_score_bins: NDArray[np.float64]
        if step is None:
            discrete_score_bins = self.discrete_score_bins
        else:
            discrete_score_bins = self._get_discrete_score_bins(step)

        discretized_score_indices = np.digitize(scores, discrete_score_bins) - 1
        return self.discrete_score_bins[discretized_score_indices], discretized_score_indices

    @staticmethod
    def _get_discrete_score_bins(score_discretization_step: float, max_precision: int = 8) -> NDArray[np.float64]:
        return np.round(
            np.arange(0, 1, score_discretization_step),
            decimals=min(int(np.ceil(-np.log10(score_discretization_step)) + 2), max_precision),
        )

    def _compute_mu_a(
        self,
        discrete_y_scores: NDArray[np.float64],
        y_ground_truth: NDArray[np.float64],
        sensitive_features: NDArray[np.float64],
    ) -> pd.DataFrame:
        """Compute mu_t^A from paper.

        This is the score distribution expected (per group) if all scores are calibrated."""

        # Get calibrated scores (known as S_A in paper)
        calibrated_scores = FAIM._compute_calibrated_scores(discrete_y_scores, y_ground_truth, sensitive_features)

        # Return distributions (index = discrete scores, columns = sensitive groups)
        return pd.DataFrame(
            data={
                sensitive_feature: self._histogram(calibrated_scores[sensitive_features == sensitive_feature])
                for sensitive_feature in np.unique(sensitive_features)
            },
            index=self.discrete_score_bins,
        )

    def _compute_mu_b_and_c(
        self,
        discrete_y_scores: NDArray[np.float64],
        discrete_y_score_indices: NDArray[np.int64],
        y_ground_truth: NDArray[np.float64],
        sensitive_features: NDArray[np.float64],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute mu_t^B and mu_t^C from paper.

        These are the group weighted expected score distributions (for each group) when negative and positive
        classes are balanced, respectively.
        """
        # Get score distributions for groups B and C (known as mu_t^[-+] in paper)
        sigma_bar_minus, sigma_bar_plus = self._compute_sigma_bar_minus_and_plus(
            discrete_y_scores, y_ground_truth, sensitive_features
        )

        mus: dict[str, dict[str, NDArray[np.float64]]] = {}
        mu_dfs: list[pd.DataFrame] = []
        for sigma_bar, ground_truth_value in zip((sigma_bar_minus, sigma_bar_plus), np.unique(y_ground_truth)):
            mus[ground_truth_value] = {}
            for sensitive_group in np.unique(sensitive_features):
                mask = (y_ground_truth == ground_truth_value) & (sensitive_features == sensitive_group)
                discrete_fair_score_map = self._get_discrete_fair_score_map(
                    scores=discrete_y_scores[mask],
                    fair_score_distribution=sigma_bar,
                )
                discrete_fair_scores = discrete_fair_score_map[discrete_y_score_indices[mask]]
                assert not np.isnan(discrete_fair_scores.sum())

                mus[ground_truth_value][sensitive_group] = self._histogram(discrete_fair_scores)

            mu_dfs.append(
                pd.DataFrame(
                    data={sensitive_group: mu for sensitive_group, mu in mus[ground_truth_value].items()},
                    index=self.discrete_score_bins,
                )
            )

        assert len(mu_dfs) == 2
        return cast(tuple[pd.DataFrame, pd.DataFrame], tuple(mu_dfs))

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

    def _compute_sigma_bar_minus_and_plus(
        self,
        discrete_y_scores: NDArray[np.float64],
        y_ground_truth: NDArray[np.float64],
        sensitive_features: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute \bar{sigma}^- and \bar{sigma}^+ from paper.

        These are the group-weighted expected score distributions when negative and positive
        classes are balanced, respectively.
        """
        # Get score distributions for each group (known as sigma_t^[-+] in paper)
        sigma_minus_and_plus_by_sensitive_group, group_counts = self._compute_sigma_minus_and_plus_by_sensitive_group(
            discrete_y_scores, y_ground_truth, sensitive_features
        )

        group_weights_by_class = group_counts.unstack() / group_counts.unstack().sum(axis=0)

        # Get idealized score distributions that balanced negative and positive classes for each group
        #  (sigma_bar^- and sigma_bar^+ in the paper, respectively)
        balanced_score_distribution_for_negative_class = ot.bregman.barycenter(
            A=sigma_minus_and_plus_by_sensitive_group.unstack("sensitive_group").loc[False].to_numpy(),
            M=self._ot_loss_matrix,
            reg=self.optimal_transport_regularization,
            weights=group_weights_by_class[False].to_numpy(),
        )
        balanced_score_distribution_for_positive_class = ot.bregman.barycenter(
            A=sigma_minus_and_plus_by_sensitive_group.unstack("sensitive_group").loc[True].to_numpy(),
            M=self._ot_loss_matrix,
            reg=self.optimal_transport_regularization,
            weights=group_weights_by_class[True].to_numpy(),
        )

        return balanced_score_distribution_for_negative_class, balanced_score_distribution_for_positive_class

    def _compute_sigma_minus_and_plus_by_sensitive_group(
        self,
        discrete_y_scores: NDArray[np.float64],
        y_ground_truth: NDArray[np.float64],
        sensitive_features: NDArray[np.float64],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Compute sigma_t^[-+] distributions from paper.

        These are the fraction of negative and positive class for each score value for each group.
        """
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
            [sensitive_groups, ground_truth_values, self.discrete_score_bins]
        )
        grouped_discretized_score_counts = (
            data.groupby(["sensitive_features", "y_ground_truth", "discrete_y_scores"], sort=True)
            .agg(score_count=("discrete_y_scores", "count"))
            .reindex(cartesion_product_of_index_features_with_discrete_score_basis)
            .fillna(0)
        )

        # Efficiently count sensitive groups associated with each class
        group_counts = grouped_discretized_score_counts.unstack([0, 1]).sum()
        group_counts.index = group_counts.index.droplevel()
        group_counts.index.names = ["sensitive_group", "ground_truth"]

        # Normalized score counts
        for group in sensitive_groups:
            for ground_truth_value in ground_truth_values:
                grouped_discretized_score_counts.loc[group, ground_truth_value] = (
                    grouped_discretized_score_counts.loc[group, ground_truth_value]
                    / grouped_discretized_score_counts.loc[group, ground_truth_value].sum()
                ).to_numpy()

        grouped_discretized_score_counts.index.names = ["sensitive_group", "ground_truth", "y_score"]
        grouped_discretized_score_counts.columns = ["fraction"]
        return grouped_discretized_score_counts, group_counts

    def _histogram(self, scores: ArrayLike, normalize: bool = False) -> NDArray[np.float64]:
        hist = np.histogram(
            scores,
            bins=np.append(self.discrete_score_bins, 1),
            density=False,
        )[0]

        if normalize:
            return hist / len(scores)
        return hist

    def _get_discrete_fair_score_map(
        self, scores: NDArray[np.float64], fair_score_distribution: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Convert scores and desired fair score distribution into a discrete fair score map.

        The discrete fair score map returned by this method is an array the size of self.discrete_score_bins
        that maps those discrete scores to fair scores based on the set of fairness criteria encoded by the
        desired "fair" score distribution.

        Parameters
        ----------
        scores:
            Scores to be mapped via Earth movers distance to fair scores
        fair_score_distribution:
            Desired fair score distribution

        Returns
        -------
        NDArray[np.float64]
            Discrete fair score map - an array the size of self.discrete_score_bins that maps discrete scores to
            fair scores


        """
        # Calculate transport map from discretized_scores to fair score distribution via Earth Movers Distance calculation
        #  ToDo: Consider using a more efficient OT solver like Sinkhorn
        transport_map = ot.emd(
            self._histogram(scores, normalize=True),
            fair_score_distribution,
            self._ot_loss_matrix,
        )

        # Normalize each row of transport ignoring rows that sum to 0
        row_sums = transport_map.sum(
            axis=1
        )  # Fraction each score value to be transported to all fair discretized_scores
        row_sums[row_sums == 0] = np.nan
        inverse_norm_vec = np.reciprocal(row_sums, where=row_sums != 0)
        norm_matrix = np.diag(inverse_norm_vec)
        normalized_transport_map = np.matmul(norm_matrix, transport_map)

        # Average resulting fair score after mapping (array aligns with input discretized_scores)
        return np.matmul(normalized_transport_map, self.discrete_score_bins.T)

    def _fill_discrete_fair_score_map(self, discrete_fair_score_map: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fill nan's in discrete fair score map.

        The discrete fair score map maps scores that have been discretized from 0 to 1
        (via the function self._get_discrete_score_values).

        When fitting FAIM, if a particular score is never seen, optimal transport will not know how to map it,
        leaving a nan-value in the discrete fair score map.  This function fills in the nans by linearly interpolating.
        If the start and/or end of the score map is nan, then it is assigned to be 0 and/or 1, respectively.
        """

        # Handle beginning and end
        is_nan = np.isnan(discrete_fair_score_map)
        if is_nan[0]:
            discrete_fair_score_map[0] = 0
        if is_nan[-1]:
            discrete_fair_score_map[-1] = 1 - self.score_discretization_step

        # Fill in remaining nans
        is_nan = np.isnan(discrete_fair_score_map)
        indices = np.arange(len(discrete_fair_score_map))
        return np.interp(indices, indices[~is_nan], discrete_fair_score_map[~is_nan])

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
