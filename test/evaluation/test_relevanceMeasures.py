"""
Created on Jun 19, 2019

@author: meike
"""
import unittest
import pandas as pd
import numpy as np
import uuid
from evaluation.relevanceMeasures import pak


class Test(unittest.TestCase):
    def setUp(self):
        self._dataSize = 20
        uuidCol = []
        for _ in np.arange(self._dataSize):
            uuidCol.append(uuid.uuid4())
        origScoreCol = np.arange(self._dataSize)
        fairScoreCol = np.arange(10, self._dataSize + 10)
        self._origData = pd.DataFrame(np.column_stack((origScoreCol, uuidCol)), columns=["score", "uuid"])
        self._fairData = pd.DataFrame(np.column_stack((fairScoreCol, uuidCol)), columns=["fairScore", "uuid"])

    def test_pak_fullyCorrectOrder(self):
        # all items in origData and fairData are ordered in the same way
        # precision at k should always be 1
        self._fairData = self._fairData.sort_values(by=["fairScore", "uuid"], ascending=[False, True])
        self._origData = self._origData.sort_values(by=["score", "uuid"], ascending=[False, True])

        fairIDs = self._fairData["uuid"].values
        origIDs = self._origData["uuid"].values

        for k in np.arange(self._dataSize):
            self.assertEqual(1, pak(k + 1, fairIDs, origIDs), "failed at k={0}".format(k))

    def test_pak_invertedOrder(self):
        # all items in origData and fairData are ordered in reverse way
        # precision at k should increase
        self._fairData = self._fairData.assign(fairScore=np.arange(self._dataSize + 10, 10, -1))
        self._fairData = self._fairData.sort_values(by=["fairScore", "uuid"], ascending=[False, True])
        self._origData = self._origData.sort_values(by=["score", "uuid"], ascending=[False, True])

        fairIDs = self._fairData["uuid"].values
        origIDs = self._origData["uuid"].values

        expectedPak = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            (2 / 11),
            (4 / 12),
            (6 / 13),
            (8 / 14),
            (10 / 15),
            (12 / 16),
            (14 / 17),
            (16 / 18),
            (18 / 19),
            (20 / 20),
        ]

        actualPak = np.empty(self._dataSize)
        for k in np.arange(self._dataSize):
            actualPak[k] = pak(k + 1, fairIDs, origIDs)

        np.testing.assert_array_equal(expectedPak, actualPak)

    def test_pak_fullyCorrectOrderSameScores(self):
        # all items in origData and fairData are ordered in the same way
        # precision at k should always be 1
        origScoreCol = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        fairScoreCol = [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
        self._origData = self._origData.assign(score=origScoreCol)
        self._fairData = self._fairData.assign(fairScore=fairScoreCol)

        self._fairData = self._fairData.sort_values(by=["fairScore", "uuid"], ascending=[False, True])
        self._origData = self._origData.sort_values(by=["score", "uuid"], ascending=[False, True])

        fairIDs = self._fairData["uuid"].values
        origIDs = self._origData["uuid"].values

        for k in np.arange(self._dataSize):
            self.assertEqual(1, pak(k + 1, fairIDs, origIDs), "failed at k={0}".format(k))

    def test_pak_invertedOrderSameScores(self):
        # all items in origData and fairData are ordered in reverse way
        # precision at k should increase
        origScoreCol = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        fairScoreCol = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25]

        self._origData = self._origData.assign(score=origScoreCol)
        self._fairData = self._fairData.assign(fairScore=fairScoreCol)

        self._fairData = self._fairData.sort_values(by=["fairScore", "uuid"], ascending=[False, True])
        self._origData = self._origData.sort_values(by=["score", "uuid"], ascending=[False, True])

        fairIDs = self._fairData["uuid"].values
        origIDs = self._origData["uuid"].values

        expectedPak = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            (2 / 11),
            (4 / 12),
            (6 / 13),
            (8 / 14),
            (10 / 15),
            (12 / 16),
            (14 / 17),
            (16 / 18),
            (18 / 19),
            (20 / 20),
        ]

        actualPak = np.empty(self._dataSize)
        for k in np.arange(self._dataSize):
            actualPak[k] = pak(k + 1, fairIDs, origIDs)

        np.testing.assert_array_equal(expectedPak, actualPak)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_pak']
    unittest.main()
