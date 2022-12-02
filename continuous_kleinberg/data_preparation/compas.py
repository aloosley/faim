"""
@author: mzehlike

protected attributes: sex, race, age
scores: decile_score

"""
import os
import pandas as pd

from continuous_kleinberg.visualization.plots import plotKDEPerGroup


class CompasCreator:
    # TODO: Do we want to run experiments with violent recidivism, too?

    def __init__(self, pathToDataFile):
        # prepare dataset as ProPublica did (see https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb)
        self.__origDataset = pd.read_csv(pathToDataFile)
        print(self.__origDataset.shape)
        self.__origDataset = self.__origDataset.drop(
            self.__origDataset[self.__origDataset.days_b_screening_arrest > 30].index
        )
        self.__origDataset = self.__origDataset.drop(
            self.__origDataset[self.__origDataset.days_b_screening_arrest < -30].index
        )
        self.__origDataset = self.__origDataset.drop(self.__origDataset[self.__origDataset.is_recid == -1].index)
        self.__origDataset = self.__origDataset.drop(
            self.__origDataset[self.__origDataset.c_charge_degree == "O"].index
        )
        self.__origDataset = self.__origDataset.drop(self.__origDataset[self.__origDataset.score_text == "N/A"].index)
        print(self.__origDataset.shape)

    def prepareGenderData(self, writingPath):
        keepCols = [
            "id",
            "sex",
            "decile_score",
            "is_recid",
            "two_year_recid",
            "c_charge_degree",
            "priors_count",
        ]
        data = self.__origDataset[keepCols].copy()

        print(data["sex"].value_counts(normalize=True))

        data["sex"] = data["sex"].replace(to_replace="Male", value=0)
        data["sex"] = data["sex"].replace(to_replace="Female", value=1)

        data = data.rename(
            columns={
                "sex": "group",
                "decile_score": "pred_score",
                "two_year_recid": "groundTruthLabel",
            }
        )
        plotKDEPerGroup(
            data,
            {0: "Male", 1: "Female"},
            "pred_score",
            os.path.join(writingPath, "dataPlot_sex.png"),
        )
        data.to_csv(os.path.join(writingPath, "data.csv"), index=False, header=True)

    def prepareRaceData(self, writingPath):
        keepCols = [
            "id",
            "race",
            "decile_score",
            "is_recid",
            "two_year_recid",
            "c_charge_degree",
            "priors_count",
        ]
        data = self.__origDataset[keepCols].copy()

        data["race"] = data["race"].replace(to_replace="Caucasian", value=0)
        data["race"] = data["race"].replace(to_replace="African-American", value=1)
        data["race"] = data["race"].replace(to_replace="Hispanic", value=2)
        # putting Asians and Native Americans into same category as Other, since there is only so few of them
        data["race"] = data["race"].replace(to_replace="Asian", value=3)
        data["race"] = data["race"].replace(to_replace="Native American", value=3)
        data["race"] = data["race"].replace(to_replace="Other", value=3)

        print(data["race"].value_counts(normalize=True))
        data = data.rename(
            columns={
                "race": "group",
                "decile_score": "pred_score",
                "two_year_recid": "groundTruthLabel",
            }
        )
        plotKDEPerGroup(
            data,
            {0: "Caucasian", 1: "Afr.-Amer.", 2: "Hispanic", 3: "Other"},
            "pred_score",
            os.path.join(writingPath, "dataPlot_race.png"),
        )
        data.to_csv(os.path.join(writingPath, "data.csv"), index=False, header=True)

    def prepareAgeData(self, writingPath):
        keepCols = [
            "id",
            "age_cat",
            "decile_score",
            "is_recid",
            "two_year_recid",
            "c_charge_degree",
            "priors_count",
        ]
        data = self.__origDataset[keepCols].copy()

        print(data["age_cat"].value_counts(normalize=True))

        data["age_cat"] = data["age_cat"].replace(to_replace="Greater than 45", value=0)
        data["age_cat"] = data["age_cat"].replace(to_replace="25 - 45", value=1)
        data["age_cat"] = data["age_cat"].replace(to_replace="Less than 25", value=2)

        data = data.rename(
            columns={
                "age_cat": "group",
                "decile_score": "pred_score",
                "two_year_recid": "groundTruthLabel",
            }
        )
        plotKDEPerGroup(
            data,
            {0: "$> 45$", 1: "$25 - 45$", 2: "$< 25$"},
            "pred_score",
            os.path.join(writingPath, "dataPlot_age.png"),
        )
        data.to_csv(os.path.join(writingPath, "data.csv"), index=False, header=True)
