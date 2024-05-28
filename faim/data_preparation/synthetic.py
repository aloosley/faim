import uuid
from pathlib import Path

import numpy as np
import pandas as pd


class SyntheticGroupedDatasetBuilder:
    def __init__(self, size: int, group_names: list[str]) -> None:
        self.size = size
        self.group_names = group_names

    def build(self) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def _init_empty_dataset(size: int, n_groups: int) -> pd.DataFrame:
        df_synth = pd.DataFrame()
        df_synth["group"] = np.random.randint(low=0, high=n_groups, size=size)
        df_synth["uuid"] = [uuid.uuid4().int for _ in range(size)]
        return df_synth


class NormalSyntheticGroupedDatasetBuilder(SyntheticGroupedDatasetBuilder):
    def __init__(self, size: int, group_names: list[str], correlation_matrix: np.ndarray[np.float32]) -> None:
        self._validate_corrleation_matrix(correlation_matrix=correlation_matrix, group_names=group_names)
        super().__init__(size=size, group_names=group_names)

    def build(self) -> pd.DataFrame:
        df_synth = self._init_empty_dataset(size=self.size, n_groups=len(self.group_names))
        return df_synth

    @staticmethod
    def _validate_corrleation_matrix(correlation_matrix: np.ndarray[np.float32], group_names: list[str]) -> None:
        if len(correlation_matrix.shape) != 2:
            raise ValueError("Correlation matrix must be 2D")

        if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
            raise ValueError("Correlation matrix must be square")

        if correlation_matrix.shape[0] != len(group_names):
            raise ValueError("Correlation matrix must have the same number of rows as group_names")


class SyntheticDatasetCreator(object):
    @property
    def dataset(self):
        return self.__dataset

    @property
    def boundary(self):
        return self.__boundary

    def __init__(self, size: int, group_count: int) -> None:
        """
        @param size:                            total number of data points to be created
        @param group_count:                     total number of groups
        """

        self.__dataset = pd.DataFrame()

        # assign groups to items
        self.__dataset["group"] = np.random.randint(0, group_count, size)

        # generate ID column with 128-bit integer IDs
        self.__dataset["uuid"] = [uuid.uuid4().int for _ in range(len(self.__dataset.index))]

    def sortByColumn(self, colName):
        self.__dataset = self.__dataset.rename_axis("idx").sort_values(by=[colName, "idx"], ascending=[False, True])

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
                y = np.random.multivariate_normal([mu1, mu2], covMatr, size=len(x))
            else:
                mu1 = 1
                mu2 = 2
                y = np.random.multivariate_normal([mu1, mu2], covMatr, size=len(x))

            x["true_score"] = y[:, 0]
            x["pred_score"] = y[:, 1]
            return x

        self.__dataset = self.__dataset.groupby(self.__dataset["group"], as_index=False, sort=False).apply(score)

    def writeToCSV(self, output_filepath: Path) -> None:
        self.__dataset.to_csv(output_filepath, index=False, header=True)
