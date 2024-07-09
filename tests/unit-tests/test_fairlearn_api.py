import numpy as np

from faim.fairlearn_api import FAIM


def test_compute_sigma_a() -> None:
    # GIVEN
    y_groundtruth = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=bool)
    discrete_y_scores = np.array([0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2, 0.2])
    sensitive_features = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    # WHEN
    sigma_a = FAIM._compute_sigma_a_scores(discrete_y_scores, y_groundtruth, sensitive_features)

    # THEN
    assert np.array_equal(sigma_a, np.array([0.5, 0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]))


def test_discretize_scores() -> None:
    # GIVEN
    y_scores = np.array([0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2, 0.2])
    step = 0.5

    faim = FAIM(thetas=[0, 0, 1], score_discretization_step=0.5)

    # WHEN
    discretized = faim._discretize_scores(y_scores)

    # THEN
    assert np.array_equal(discretized, np.array([0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0]))
