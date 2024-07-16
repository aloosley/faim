import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal

from faim.fairlearn_api import FAIM


class TestFAIM:
    def test_fit(self) -> None:
        # GIVEN a FAIM instance
        thetas = {0: [0, 0, 1], 1: [1, 0, 0]}
        score_discretization_step = 0.007194244604316547  # Step value in original code after normalizing data

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
        faim = faim.fit(y_scores, y_ground_truth, sensitive_features=sensitive_features)

        # THEN FAIM type is returned
        assert isinstance(faim, FAIM)

        # THEN the score maps are roughly equal (unfortunately they vary slightly from platform to platfrom)
        assert isinstance(faim.discrete_fair_score_map_by_group, dict)

        rtol_score_elements = 1e-3
        rtol_mean_score = rtol_score_elements / np.sqrt(len(faim.discrete_fair_score_map_by_group[0]))

        assert np.isclose(faim.discrete_fair_score_map_by_group[0].mean(), 0.61148966, rtol=rtol_mean_score)
        assert np.isclose(faim.discrete_fair_score_map_by_group[1].mean(), 0.30627379, rtol=rtol_mean_score)

        assert np.allclose(
            faim.discrete_fair_score_map_by_group[0][:5],
            np.array([0.333111, 0.338481, 0.343851, 0.349222, 0.354592]),
            rtol=rtol_score_elements,
        )
        assert np.allclose(
            faim.discrete_fair_score_map_by_group[0][-5:],
            np.array([0.82734, 0.83122597, 0.83812917, 0.84419487, 0.85026057]),
            rtol=rtol_score_elements,
        )
        assert np.allclose(
            faim.discrete_fair_score_map_by_group[1][:5],
            np.array([0.040875, 0.042148, 0.043421, 0.044694, 0.045967]),
            rtol=rtol_score_elements,
        )
        assert np.allclose(
            faim.discrete_fair_score_map_by_group[1][-5:],
            np.array([0.9299756, 0.93525, 0.93820437, 0.94205291, 0.94590145]),
            rtol=rtol_score_elements,
        )

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
        discrete_y_score_indices = np.array([1, 1, 2, 3, 3, 3, 3, 3, 1, 1])
        sensitive_features = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        score_discretization_step = 0.2

        faim = FAIM(thetas=[0, 0, 1], score_discretization_step=score_discretization_step)

        # WHEN
        mu_b, mu_c = faim._compute_mu_b_and_c(
            discrete_y_scores=discrete_y_scores,
            discrete_y_score_indices=discrete_y_score_indices,
            y_ground_truth=y_ground_truth,
            sensitive_features=sensitive_features,
        )

        # THEN
        assert_frame_equal(
            mu_b, pd.DataFrame(data={0: [0, 1, 0, 1, 0], 1: [0, 2, 0, 2, 0]}, index=[0, 0.2, 0.4, 0.6, 0.8])
        )
        assert_frame_equal(
            mu_c, pd.DataFrame(data={0: [0, 0, 2, 0, 0], 1: [0, 0, 2, 0, 0]}, index=[0, 0.2, 0.4, 0.6, 0.8])
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
        assert np.array_equal(
            np.squeeze(sigma_minus_and_plus_by_group.loc[0, 1].to_numpy()),
            np.array([0, 1 / 3, 1 / 3, 1 / 3, 0]),
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
        assert len(sigma_bar_minus) == len(sigma_bar_plus) == len(faim.discrete_score_bins)

        # THEN distributions are (almost) normalized
        np.testing.assert_approx_equal(sigma_bar_minus.sum(), sigma_bar_plus.sum())
        np.testing.assert_approx_equal(sigma_bar_minus.sum(), 1)

        # The mean score with balanced positive class > mean score with balanced negative class
        #  (as should be by data provided)
        assert (sigma_bar_minus * np.arange(len(sigma_bar_minus))).sum() < (
            sigma_bar_plus * np.arange(len(sigma_bar_minus))
        ).sum()

        # THEN fixed value expected
        assert np.allclose(
            sigma_bar_minus,
            np.array([3.593892e-28, 5.000001e-01, 2.500000e-01, 2.499999e-01, 1.795358e-47]),
        )
        assert np.allclose(
            sigma_bar_plus,
            np.array([4.596714e-164, 1.722138e-055, 3.333329e-001, 6.666671e-001, 4.791857e-028]),
        )

    def test_discrete_score_bins(self) -> None:
        # GIVEN
        score_discretization_step = 0.2

        # WHEN
        faim = FAIM(thetas=[0, 0, 1], score_discretization_step=score_discretization_step)

        # THEN
        assert np.array_equal(faim.discrete_score_bins, np.array([0.0, 0.2, 0.4, 0.6, 0.8]))

    def test_discretize_scores(self) -> None:
        # GIVEN
        y_scores = np.array([0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2, 0.2, 0.9999999, 0.5, 1.0])
        step = 0.5

        faim = FAIM(thetas=[0, 0, 1], score_discretization_step=step)

        # WHEN
        discretized_y_scores, discretized_y_score_indices = faim._get_discrete_scores_and_indices(y_scores)

        # THEN
        assert np.array_equal(discretized_y_scores, np.array([0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0.5, 0.5, 0.5]))
        assert np.array_equal(discretized_y_score_indices, np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]))

    def test_fill_discrete_fair_score_map(self) -> None:
        # GIVEN
        score_map = np.array([np.nan, 0.1, 0.4, 0.6, 0.6, np.nan, 0.7, 0.8, 0.9, np.nan])
        faim = FAIM(thetas=[0, 0, 1], score_discretization_step=0.01)

        # WHEN
        interpolated_score_map = faim._fill_discrete_fair_score_map(score_map)

        # THEN
        assert np.allclose(interpolated_score_map, np.array([0.0, 0.1, 0.4, 0.6, 0.6, 0.65, 0.7, 0.8, 0.9, 0.99]))
