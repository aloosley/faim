"""
Created on Sep 28, 2018

@author: mzehlike
"""
import unittest
import pandas as pd
import numpy as np
from copy import deepcopy
from algorithm import fia
import pytest

DATA = [
    [1, 1, 9, 0, ""],
    [0, 1, 8, 0, ""],
    [1, 1, 7, 1, ""],
    [0, 1, 6, 1, ""],
    [1, 0, 5, 0, ""],
    [0, 0, 4, 0, ""],
    [1, 0, 3, 1, ""],
    [0, 0, 2, 1, ""],
]


@pytest.fixture
def fia_object():
    return get_fia_object_from_data(DATA)


def get_fia_object_from_data(data):
    cols = ["groundTruthLabel", "predictedLabel", "predictedScore", "group", "uuid"]
    df = pd.DataFrame(data, columns=cols)
    return fia.FairnessInterpolationAlgorithm(
        rawData=df,
        group_names={0: "male", 1: "female"},
        pred_score="predictedScore",
        score_stepsize=1.0,
        thetas=[1, 1, 1],
        regForOT=5e-3,
        path="",
        plot=False,
    )


@pytest.fixture
def fia_shared_scores():
    """A dataset where some group members share the same score."""
    new_data = deepcopy(DATA)
    shared_scores = [9, 9, 3, 2, 9, 8, 3, 2]
    for i, s in enumerate(shared_scores):
        new_data[i][2] = s
    return get_fia_object_from_data(new_data)


@pytest.mark.new
def test_compute_SA_score(fia_shared_scores):
    """Check that the condition A scores are computed correctly."""
    fia_shared_scores._calculate_muA_perGroup()
    data = fia_shared_scores._FairnessInterpolationAlgorithm__data
    #                   0  1  2  3  4  5  6  7
    # groundTruthLabel  1  0  1  0  1  0  1  0
    # predictedScore    9  9  3  2  9  8  3  2
    # group             0  0  1  1  0  0  1  1
    expected = pd.Series([2 / 3, 2 / 3, 1.0, 0.0, 2 / 3, 0.0, 1.0, 0.0], index=data.index)
    pd.testing.assert_series_equal(data[fia.SA_COLNAME], expected, check_names=False)


@pytest.mark.new
def test_dataToHistograms(fia_object):
    data = fia_object._FairnessInterpolationAlgorithm__data
    mn, mx = data["predictedScore"].min(), data["predictedScore"].max()
    n = np.nan
    df = (pd.DataFrame([[2, 5, n], [3, 7, n], [4, n, n], [5, -2, n]]) - mn) / (mx - mn)
    index = np.linspace(0, 1, fia_object._FairnessInterpolationAlgorithm__numBins + 1)[:-1]
    h1, h2, h3 = [1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]
    expected = pd.DataFrame(data=list(zip(h1, h2, h3)), columns=[0, 1, 2], index=index)
    res = fia_object._dataToHistograms(df)
    pd.testing.assert_frame_equal(res, expected)


@pytest.mark.new
def test_calculateFairReplacementStrategy(fia_object):
    n = np.nan
    raw_scores = pd.DataFrame(
        {
            0: [0.1428, 0.2142, 0.2857, 0.3573, n, n, n, n],
            1: [0.0434, 0.0652, 0.0869, 0.1308, 0.1521, 0.1521, 0.1739, 0.1956],
        }
    )
    grp_bary = pd.DataFrame(
        {
            0: [0.2222, 0.2407, 0.2594, 0.2777, n, n, n, n],
            1: [0.1067, 0.1116, 0.1169, 0.1262, 0.1310, 0.1310, 0.1359, 0.1407],
        }
    )
    expected = pd.DataFrame(
        {
            0: [0.0, 0.0786648, 0.20366643, 0.34715225, n, n, n, n],
            1: [
                0.0,
                0.00364264,
                0.125,
                0.22821101,
                0.3816568,
                0.53131164,
                0.68322312,
                0.83991564,
            ],
        }
    )
    res = fia_object._calculateFairReplacementStrategy(grp_bary, raw_scores)
    pd.testing.assert_frame_equal(res, expected)


class Test_FairnessInterpolationAlgorithm(unittest.TestCase):
    # TODO: refactoring needed for all tests, but done for setup()
    def setUp(self):
        self.scoreAttr = "predicted_score"
        datapoints = np.array(
            [
                [int(1), int(1), int(9), int(0), ""],
                [int(0), int(1), int(8), int(0), ""],
                [int(1), int(1), int(7), int(1), ""],
                [int(0), int(1), int(6), int(1), ""],
                [int(1), int(0), int(5), int(0), ""],
                [int(0), int(0), int(4), int(0), ""],
                [int(1), int(0), int(3), int(1), ""],
                [int(0), int(0), int(2), int(1), ""],
            ],
            dtype=object,
        )
        self.scoreRanges = [2, 9]
        self.scoreStepsize = 1
        self.regForOT = 5e-3
        self.groupNames = {0: "male", 1: "female"}
        self.smallRawData = pd.DataFrame(
            datapoints, columns=["groundTruthLabel", "predictedLabel", "predictedScore", "group", "uuid"]
        )

    def tearDown(self):
        pass

    def test_getScoresByGroup_SameGroupSizes(self):
        fia_object = fia.FairnessInterpolationAlgorithm(
            self.smallRawData,
            self.groupNames,
            "predictedScore",
            self.scoreStepsize,
            [1, 1, 1],
            self.regForOT,
            path="",
            plot=False,
        )
        expectedData = np.array([[9, 7, 5, 3], [8, 6, 4, 2]], dtype=object)
        expected = pd.DataFrame(expectedData, columns=["[0 0]", "[0 1]", "[1 0]", "[1 1]"])
        actual = fia_object._getScoresByGroup(self.smallRawData)
        pd.testing.assert_frame_equal(expected, actual)

    def test_getScoresByGroup_DifferentGroupSizes(self):
        extraRow = pd.DataFrame([[int(1), int(1), int(1), ""]], columns=["gender", "ethnicity", "score", "uuid"])
        self.smallRawData = self.smallRawData.append(extraRow)
        fia_object = fia.FairnessInterpolationAlgorithm(
            self.smallRawData,
            self.groupNames,
            "predictedScore",
            self.scoreStepsize,
            [1, 1, 1],
            self.regForOT,
            path="",
            plot=False,
        )
        expectedData = np.array([[9, 7, 5, 3], [8, 6, 4, 2], [np.NaN, np.NaN, np.NaN, 1]], dtype=object)
        expected = pd.DataFrame(expectedData, columns=["[0 0]", "[0 1]", "[1 0]", "[1 1]"])
        actual = fia_object._getScoresByGroup(self.smallRawData)
        pd.testing.assert_frame_equal(expected, actual)

    def test_calculate_SA_perGroup(self):
        self.smallRawData


if __name__ == "__main__":
    unittest.main()
