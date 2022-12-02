"""
Created on Sep 28, 2018

@author: mzehlike
"""
import pytest
import unittest
import pandas as pd
import numpy as np
from util import util


class Test(unittest.TestCase):
    def setUp(self):
        cols = ["group", "score"]
        self.dataset = pd.DataFrame([[0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [1, 2], [1, 2]], columns=cols)
        self.groups = sorted(self.dataset.group.unique())

    def test_scoresByGroups(self):
        scoresPerGroup = util.scoresByGroup(self.dataset, self.groups, "score")
        expected = pd.DataFrame([[10, 2], [10, 2], [10, np.NAN], [10, np.NAN], [10, np.NAN]], columns=[0, 1])
        pd.testing.assert_frame_equal(expected, scoresPerGroup)

    def test_scoresByGroups_empty_group(self):
        scoresPerGroup = util.scoresByGroup(self.dataset, [0, 1, 2], "score")
        expected = pd.DataFrame(
            # fmt: off
            [
                [10, 2,      np.nan],
                [10, 2,      np.nan],
                [10, np.nan, np.nan],
                [10, np.nan, np.nan],
                [10, np.nan, np.nan],
            ],
            # fmt: on
            columns=[0, 1, 2],
        )
        pd.testing.assert_frame_equal(expected, scoresPerGroup)


@pytest.mark.new
def test_normalizeRowsToOne():
    original = np.array(
        [
            [1, 2, 3],
            [-1, 2, 0],
            [5, 3, 2.5],
        ]
    )
    expected = np.array(
        [
            [1 / 6, 2 / 6, 3 / 6],
            [-1, 2, 0],
            [5 / 10.5, 3 / 10.5, 2.5 / 10.5],
        ]
    )
    res = util.normalizeRowsToOne(original)
    np.testing.assert_allclose(res, expected)


@pytest.mark.new
def test_normalizeRowsToOne_with_zero_row():
    original = np.array(
        [
            [1, 2, 3],
            [0, 0, 0],
            [1, 2, 3],
        ]
    )
    expected = np.array(
        [
            [1 / 6, 2 / 6, 3 / 6],
            [0, 0, 0],
            [1 / 6, 2 / 6, 3 / 6],
        ]
    )
    res = util.normalizeRowsToOne(original)
    np.testing.assert_allclose(res, expected)
