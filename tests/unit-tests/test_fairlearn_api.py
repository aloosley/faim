import numpy as np
import pandas as pd
from pandas import Index
from pandas._testing import assert_frame_equal

from faim.fairlearn_api import FAIM


class TestFAIM:
    def test_fit(self) -> None:
        # GIVEN
        y_groundtruth = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
        y_scores = np.array([0.12, 0.25, 0.385, 0.452, 0.562, 0.612, 0.422, 0.151, 0.96, 0.242])
        sensitive_features = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        faim = FAIM(thetas=[0, 0, 1], score_discretization_step=0.2)

        # WHEN
        faim.fit(y_scores, y_groundtruth, sensitive_features=sensitive_features)

        # THEN
        assert faim

    def test_compute_calibrated_scores(self) -> None:
        # GIVEN
        y_groundtruth = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=bool)
        discrete_y_scores = np.array([0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2, 0.2])
        sensitive_features = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        # WHEN
        sigma_a = FAIM._compute_calibrated_scores(discrete_y_scores, y_groundtruth, sensitive_features)

        # THEN
        assert np.array_equal(sigma_a, np.array([0.5, 0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]))

    def test_compute_mu_a(self) -> None:
        # GIVEN
        y_groundtruth = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=bool)
        discrete_y_scores = np.array([0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2, 0.2])
        sensitive_features = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        score_discretization_step = 0.2

        faim = FAIM(thetas=[0, 0, 1], score_discretization_step=score_discretization_step)

        # WHEN
        mu_a = faim._compute_mu_a(discrete_y_scores, y_groundtruth, sensitive_features)

        # THEN
        assert_frame_equal(
            mu_a,
            pd.DataFrame(
                data={0: [1, 0, 2, 0, 1], 1: [2, 0, 4, 0, 0]},
                index=[0, 0.2, 0.4, 0.6, 0.8],
            ),
            atol=1e-8,
        )

    def test_compute_sigma_b_and_c_score_distributions(self) -> None:
        # GIVEN
        y_groundtruth = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
        discrete_y_scores = np.array([0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2, 0.2])
        sensitive_features = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        score_discretization_step = 0.2

        faim = FAIM(thetas=[0, 0, 1], score_discretization_step=score_discretization_step)

        # WHEN
        grouped_score_distributions = faim._compute_sigma_b_and_c_score_distributions(
            discrete_y_scores, y_groundtruth, sensitive_features
        )

        # THEN
        assert np.array_equal(
            np.squeeze(grouped_score_distributions.loc[0, 0].to_numpy()), np.array([0, 1.0, 0, 0, 0, 0])
        )
        np.testing.assert_array_almost_equal(
            np.squeeze(grouped_score_distributions.loc[0, 1].to_numpy()),
            np.array([0, 0.33, 0.33, 0.33, 0, 0]),
            decimal=2,
        )
        assert np.array_equal(
            np.squeeze(grouped_score_distributions.loc[1, 0].to_numpy()), np.array([0, 0.5, 0, 0.5, 0, 0])
        )
        assert np.array_equal(
            np.squeeze(grouped_score_distributions.loc[1, 1].to_numpy()), np.array([0, 0, 0, 1.0, 0, 0])
        )

    def test_compute_sigma_bar_minus_and_plus(self) -> None:
        # GIVEN
        y_groundtruth = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
        discrete_y_scores = np.array([0.6, 0.2, 0.4, 0.6, 0.6, 0.8, 1.0, 0.6, 0.2, 0.2])
        sensitive_features = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        score_discretization_step = 0.2

        faim = FAIM(thetas=[0, 0, 1], score_discretization_step=score_discretization_step)

        # WHEN
        sigma_bar_minus, sigma_bar_plus = faim._compute_sigma_bar_minus_and_plus(
            discrete_y_scores, y_groundtruth, sensitive_features
        )

        # THEN output has the right shape
        assert len(sigma_bar_minus) == len(sigma_bar_plus) == len(faim.normalized_discrete_score_values)

        # THEN distributions are (almost) normalized
        np.testing.assert_approx_equal(sigma_bar_minus.sum(), sigma_bar_plus.sum())
        np.testing.assert_approx_equal(sigma_bar_minus.sum(), 1)

        # The mean score with balanced positive class > mean score with balanced negative class
        #  (as should be by data provided)
        assert (sigma_bar_minus * np.arange(len(sigma_bar_minus))).sum() < (
            sigma_bar_plus * np.arange(len(sigma_bar_minus))
        ).sum()

        # THEN fixed value expected
        np.testing.assert_array_almost_equal(
            sigma_bar_minus,
            np.array([8.97865620e-3, 4.90227959e-1, 2.88368192e-1, 2.12319076e-1, 1.06115367e-4, 1.19633925e-11]),
        )
        np.testing.assert_array_almost_equal(
            sigma_bar_plus,
            np.array([6.07960928e-12, 5.40241487e-5, 1.61645009e-1, 3.38221254e-1, 4.90463861e-1, 9.61585066e-3]),
        )

    def test_normalized_discrete_score_values(self) -> None:
        # GIVEN
        score_discretization_step = 0.2

        # WHEN
        faim = FAIM(thetas=[0, 0, 1], score_discretization_step=score_discretization_step)

        # THEN
        assert np.array_equal(faim.normalized_discrete_score_values, np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))

    def test_discretize_scores(self) -> None:
        # GIVEN
        y_scores = np.array([0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2, 0.2])
        step = 0.5

        faim = FAIM(thetas=[0, 0, 1], score_discretization_step=step)

        # WHEN
        discretized = faim._discretize_scores(y_scores)

        # THEN
        assert np.array_equal(discretized, np.array([0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0]))
