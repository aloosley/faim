import uuid
from pathlib import Path
from random import seed
from typing import Optional, List

import numpy as np
import pandas as pd
from numpy._typing import NDArray
from numpy.random import Generator


class SyntheticGroupedDatasetBuilder:
    """Clean dataset builder that should eventually replace the "Creator" below."""

    def __init__(
        self, group_names: List[str], n_by_group: List[int], random_generator: Optional[Generator] = None
    ) -> None:
        if len(group_names) != len(n_by_group):
            raise ValueError("group_names and n_by_group must have the same length")

        self.group_names = group_names
        self.n_by_group = n_by_group

        if random_generator is None:
            random_generator = np.random.default_rng()
        self.random_generator = random_generator

    def build(self) -> pd.DataFrame:
        raise NotImplementedError

    def _init_empty_dataset(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["uuid", "group", "true_score", "pred_score"])


class NormalSyntheticGroupedDatasetBuilder(SyntheticGroupedDatasetBuilder):
    def __init__(
        self,
        group_names: List[str],
        n_by_group: List[int],
        truth_prediction_means_by_group: List[NDArray[np.float32]],
        truth_prediction_correlation_matrixs_by_group: List[NDArray[np.float32]],
        random_generator: Optional[Generator] = None,
    ) -> None:
        super().__init__(n_by_group=n_by_group, group_names=group_names, random_generator=random_generator)

        self._validate_means_and_corrleations(
            truth_prediction_means_by_group=truth_prediction_means_by_group,
            truth_prediction_correlation_matrixs_by_group=truth_prediction_correlation_matrixs_by_group,
            group_names=group_names,
        )

        self.truth_prediction_means_by_group = truth_prediction_means_by_group
        self.truth_prediction_correlation_matrixs_by_group = truth_prediction_correlation_matrixs_by_group

    def build(self) -> pd.DataFrame:
        synth_data = self._init_empty_dataset()

        for group_idx, (means, correlation_matrix, n) in enumerate(
            zip(
                self.truth_prediction_means_by_group,
                self.truth_prediction_correlation_matrixs_by_group,
                self.n_by_group,
            )
        ):
            snyth_data_group = pd.DataFrame(
                self.random_generator.multivariate_normal(means, correlation_matrix, size=n),
                columns=["true_score", "pred_score"],
            )
            snyth_data_group["group"] = group_idx
            snyth_data_group["uuid"] = [uuid.uuid4().int for _ in range(n)]

            synth_data = pd.concat([synth_data, snyth_data_group])

        # Shuffle
        synth_data = synth_data.sample(frac=1, replace=False, random_state=self.random_generator)

        # Assign labels
        boundary = synth_data["true_score"].mean()
        synth_data["true_label"] = 0
        synth_data.loc[synth_data["true_score"] >= boundary, "true_label"] = 1
        synth_data["pred_label"] = 0
        synth_data.loc[synth_data["pred_score"] >= boundary, "pred_label"] = 1

        return synth_data

    @staticmethod
    def _validate_means_and_corrleations(
        truth_prediction_means_by_group: List[NDArray[np.float32]],
        truth_prediction_correlation_matrixs_by_group: List[NDArray[np.float32]],
        group_names: List[str],
    ) -> None:
        if len(truth_prediction_means_by_group) != len(truth_prediction_correlation_matrixs_by_group):
            raise ValueError(
                "truth_prediction_means_by_group and truth_prediction_correlation_matrixs_by_group must have the same length (number of groups)"
            )

        if len(truth_prediction_means_by_group) != len(group_names):
            raise ValueError("truth_prediction_means_by_group must have one entry per group")

        for means, correlation_matrix in zip(
            truth_prediction_means_by_group, truth_prediction_correlation_matrixs_by_group
        ):
            if len(means.shape) != 1:
                raise ValueError("Means must be 1D")

            if means.shape[0] != 2:
                raise ValueError("Means must be of length 2, one for true_score and one for pred_score")

            if len(correlation_matrix.shape) != 2:
                raise ValueError("Correlation matrix must be 2D")

            if correlation_matrix.shape[0] != 2:
                raise ValueError("Correlation matrices must be 2x2")


class SyntheticDatasetCreator(object):
    @property
    def dataset(self):
        return self.__dataset

    @property
    def boundary(self):
        return self.__boundary

    def __init__(self, size: int, group_count: int, random_generator: Optional[Generator] = None) -> None:
        """
        @param size:                            total number of data points to be created
        @param group_count:                     total number of groups
        """

        self.__dataset = pd.DataFrame()

        # assign groups to items
        self.__dataset["group"] = np.random.randint(0, group_count, size)

        # generate ID column with 128-bit integer IDs
        self.__dataset["uuid"] = [uuid.uuid4().int for _ in range(len(self.__dataset.index))]

        if random_generator is None:
            random_generator = np.random.default_rng()
        self.random_generator = random_generator

    def sortByColumn(self, colName):
        dataset = self.__dataset
        dataset["idx"] = dataset.index
        self.__dataset = dataset.sort_values(by=[colName, "idx"], ascending=[False, True])

    def setDecisionBoundaryAsMean(self, trueScoreCol, predScoreCol):
        boundary = self.__dataset[trueScoreCol].mean()
        self.__dataset["groundTruthLabel"] = 0
        self.__dataset.loc[self.__dataset[trueScoreCol] >= boundary, "groundTruthLabel"] = 1
        self.__dataset["predictedLabel"] = 0
        self.__dataset.loc[self.__dataset[predScoreCol] >= boundary, "predictedLabel"] = 1
        self.__boundary = boundary

    def createTwoCorrelatedNormalDistributionScores(self):
        """
        For each group in dataset
        creates two normal distributions true_score and pred_score, such that they are related
        i.e. a high true_score corresponds to a higher pred_score and a low true_score
        will have a lower pred_score
        """

        def score(x):
            covMatr = np.array([[1, 0.8], [0.8, 1]])
            # mu1 = np.random.uniform(low=-1.0)
            # mu2 = np.random.uniform(low=-4.0)
            # y = np.random.multivariate_normal([mu1, mu2], covMatr, size=len(x))

            # the following is to create a very specific setting in the data,
            # can be deleted after experiments and replaced with above commented code
            group = x["group"].iloc[0]
            if group:
                mu1 = -1
                mu2 = -3
                y = self.random_generator.multivariate_normal([mu1, mu2], covMatr, size=len(x))
            else:
                mu1 = 1
                mu2 = 2
                y = self.random_generator.multivariate_normal([mu1, mu2], covMatr, size=len(x))

            x["true_score"] = y[:, 0]
            x["pred_score"] = y[:, 1]
            return x

        self.__dataset = self.__dataset.groupby(["group"], as_index=False, sort=False, group_keys=False).apply(score)

    def writeToCSV(self, output_filepath: Path) -> None:
        self.__dataset.to_csv(output_filepath, index=False, header=True)
