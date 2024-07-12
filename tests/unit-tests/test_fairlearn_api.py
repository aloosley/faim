import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal

from faim.fairlearn_api import FAIM


class TestFAIM:
    def test_fit(self) -> None:
        # GIVEN a FAIM instance
        thetas = {0: [0, 0, 1], 1: [1, 0, 0]}
        score_discretization_step = 0.007194244604316547  # Step value in Meike's original code after normalizing data

        faim = FAIM(thetas=thetas, score_discretization_step=score_discretization_step)

        # GIVEN some data
        synthetic_data_from_paper = pd.read_csv(
            "https://raw.githubusercontent.com/MilkaLichtblau/faim/main/data/synthetic/2groups/2022-01-12/dataset.csv"
        )

        y_scores = synthetic_data_from_paper.pred_score
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
        y_ground_truth = synthetic_data_from_paper.groundTruthLabel.astype(bool)
        sensitive_features = synthetic_data_from_paper.group

        # WHEN
        faim.fit(y_scores, y_ground_truth, sensitive_features=sensitive_features)

        # THEN
        assert isinstance(faim, FAIM)

    def test_compute_calibrated_scores(self) -> None:
        # GIVEN
        y_ground_truth = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=bool)
        discrete_y_scores = np.array([0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2, 0.2])
        sensitive_features = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        # WHEN
        sigma_a = FAIM._compute_calibrated_scores(discrete_y_scores, y_ground_truth, sensitive_features)

        # THEN
        assert np.array_equal(sigma_a, np.array([0.5, 0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]))

    def test_compute_mu_a(self) -> None:
        # GIVEN
        y_ground_truth = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=bool)
        discrete_y_scores = np.array([0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2, 0.2])
        sensitive_features = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        score_discretization_step = 0.2

        faim = FAIM(thetas=[0, 0, 1], score_discretization_step=score_discretization_step)

        # WHEN
        mu_a = faim._compute_mu_a(discrete_y_scores, y_ground_truth, sensitive_features)

        # THEN
        assert_frame_equal(
            mu_a,
            pd.DataFrame(
                data={0: [1, 0, 2, 0, 1], 1: [2, 0, 4, 0, 0]},
                index=[0, 0.2, 0.4, 0.6, 0.8],
            ),
            atol=1e-8,
        )

    def test_compute_mu_b_and_c(self) -> None:
        # GIVEN
        y_ground_truth = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=bool)
        discrete_y_scores = np.array([0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2, 0.2])
        sensitive_features = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        score_discretization_step = 0.2

        faim = FAIM(thetas=[0, 0, 1], score_discretization_step=score_discretization_step)

        # WHEN
        mu_b, mu_c = faim._compute_mu_b_and_c(discrete_y_scores, y_ground_truth, sensitive_features)

        # THEN
        assert_frame_equal(
            mu_b, pd.DataFrame(data={0: [0, 0.5, 0, 0.5, 0], 1: [0, 0.5, 0, 0.5, 0]}, index=[0, 0.2, 0.4, 0.6, 0.8])
        )
        assert_frame_equal(
            mu_c, pd.DataFrame(data={0: [0.5, 0.25, 0.25, 0, 0], 1: [0.5, 0, 0.5, 0, 0]}, index=[0, 0.2, 0.4, 0.6, 0.8])
        )

    def test_compute_sigma_minus_and_plus_by_sensitive_group(self) -> None:
        # GIVEN
        y_ground_truth = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
        discrete_y_scores = np.array([0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2, 0.2])
        sensitive_features = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        score_discretization_step = 0.2

        faim = FAIM(thetas=[0, 0, 1], score_discretization_step=score_discretization_step)

        # WHEN
        sigma_minus_and_plus_by_group, group_counts = faim._compute_sigma_minus_and_plus_by_sensitive_group(
            discrete_y_scores, y_ground_truth, sensitive_features
        )

        # THEN sigmas are correct
        assert np.array_equal(
            np.squeeze(sigma_minus_and_plus_by_group.loc[0, 0].to_numpy()), np.array([0, 1.0, 0, 0, 0])
        )
        np.testing.assert_array_almost_equal(
            np.squeeze(sigma_minus_and_plus_by_group.loc[0, 1].to_numpy()),
            np.array([0, 0.33, 0.33, 0.33, 0]),
            decimal=2,
        )
        assert np.array_equal(
            np.squeeze(sigma_minus_and_plus_by_group.loc[1, 0].to_numpy()), np.array([0, 0.5, 0, 0.5, 0])
        )
        assert np.array_equal(
            np.squeeze(sigma_minus_and_plus_by_group.loc[1, 1].to_numpy()), np.array([0, 0, 0, 1.0, 0])
        )

        # THEN group counts are correct
        assert np.array_equal(np.squeeze(group_counts.to_numpy()), np.array([1, 3, 4, 2]))

    def test_compute_sigma_bar_minus_and_plus(self) -> None:
        # GIVEN
        y_ground_truth = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
        discrete_y_scores = np.array([0.6, 0.2, 0.4, 0.6, 0.6, 0.8, 1.0, 0.6, 0.2, 0.2])
        sensitive_features = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        score_discretization_step = 0.2

        faim = FAIM(thetas=[0, 0, 1], score_discretization_step=score_discretization_step)

        # WHEN
        sigma_bar_minus, sigma_bar_plus = faim._compute_sigma_bar_minus_and_plus(
            discrete_y_scores, y_ground_truth, sensitive_features
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
            np.array([9.633956e-04, 4.990513e-01, 2.711831e-01, 2.288006e-01, 1.592632e-06]),
        )
        np.testing.assert_array_almost_equal(
            sigma_bar_plus,
            np.array([1.019634e-17, 7.341867e-07, 1.970870e-01, 8.014763e-01, 1.435954e-03]),
        )

    def test_normalized_discrete_score_values(self) -> None:
        # GIVEN
        score_discretization_step = 0.2

        # WHEN
        faim = FAIM(thetas=[0, 0, 1], score_discretization_step=score_discretization_step)

        # THEN
        assert np.array_equal(faim.normalized_discrete_score_values, np.array([0.0, 0.2, 0.4, 0.6, 0.8]))

    def test_discretize_scores(self) -> None:
        # GIVEN
        y_scores = np.array([0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2, 0.2, 0.9999999, 0.5, 1.0])
        step = 0.5

        faim = FAIM(thetas=[0, 0, 1], score_discretization_step=step)

        # WHEN
        discretized_y_scores = faim._discretize_scores(y_scores)

        # THEN
        assert np.array_equal(discretized_y_scores, np.array([0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0.5, 0.5, 0.5]))
