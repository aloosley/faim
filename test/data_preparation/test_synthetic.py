import unittest

from faim.data_preparation import synthetic


class TestSyntheticDatasetCreator(unittest.TestCase):
    def setUpClass(self):
        self.nonProtectedAttributes = ["score"]
        self.protectedAttributes = {"gender": 2, "ethnicity": 3}
        self.size = 60

    def tearDown(self):
        pass

    def test_Constructor(self):
        creator = synthetic.SyntheticDatasetCreator(self.size, self.protectedAttributes, self.nonProtectedAttributes)
        self.assertTrue("gender" in creator.dataset.columns)
        self.assertTrue("ethnicity" in creator.dataset.columns)

        expectedGroups = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        self.assertCountEqual(expectedGroups, creator.groups)

    def test_createScoresNormalDistribution(self):
        creator = synthetic.SyntheticDatasetCreator(self.size, self.protectedAttributes, self.nonProtectedAttributes)

        creator.createScoresNormalDistribution(self.nonProtectedAttributes)
        self.assertTrue("score" in creator.dataset.columns)
        self.assertEqual(["score"], creator.nonProtectedAttributes)
        self.assertEqual(["gender", "ethnicity"], list(creator.protectedAttributes))


if __name__ == "__main__":
    unittest.main()
