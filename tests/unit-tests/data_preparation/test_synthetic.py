import numpy as np
from numpy.random import PCG64, Generator

from faim.data_preparation.synthetic import NormalSyntheticGroupedDatasetBuilder


class TestNormalSyntheticGroupedDatasetBuilder:
    def test_build(self) -> None:
        # GIVEN a builder
        n_by_group = [10000, 10000]
        random_generator = Generator(PCG64(4))
        synth_data_builder = NormalSyntheticGroupedDatasetBuilder(
            group_names=["advantaged", "disadvantaged"],
            n_by_group=n_by_group,
            truth_prediction_means_by_group=[np.array([-1, -3]), np.array([1, 2])],
            truth_prediction_correlation_matrixs_by_group=[
                np.array([[1, 0.8], [0.8, 1]]),
                np.array([[1, 0.8], [0.8, 1]]),
            ],
            random_generator=random_generator,
        )

        # WHEN building the synthetic data
        synth_data = synth_data_builder.build()

        # THEN
        assert synth_data.shape == (sum(n_by_group), 6)
        assert list(synth_data.columns) == ["uuid", "group", "true_score", "pred_score", "true_label", "pred_label"]
        assert synth_data["group"].nunique() == 2
        assert synth_data.iloc[4, 1] == 1
