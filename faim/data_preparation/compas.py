"""
@author: mzehlike

protected attributes: sex, race, age
scores: decile_score

"""
from pathlib import Path

import pandas as pd

from faim.visualization.plots import plotKDEPerGroup


class CompasCreator:
    # TODO: Do we want to run experiments with violent recidivism, too?

    def __init__(self, data_filepath: Path, output_dir: Path) -> None:
        self.data_filepath = data_filepath
        self.output_dir = output_dir

        # prepare dataset as ProPublica did (see https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb)
        self.__dataset: pd.DataFrame = pd.read_csv(data_filepath)
        print(self.__dataset.shape)
        self.__dataset = self.__dataset.drop(self.__dataset[self.__dataset.days_b_screening_arrest > 30].index)
        self.__dataset = self.__dataset.drop(self.__dataset[self.__dataset.days_b_screening_arrest < -30].index)
        self.__dataset = self.__dataset.drop(self.__dataset[self.__dataset.is_recid == -1].index)
        self.__dataset = self.__dataset.drop(self.__dataset[self.__dataset.c_charge_degree == "O"].index)
        self.__dataset = self.__dataset.drop(self.__dataset[self.__dataset.score_text == "N/A"].index)
        print(self.__dataset.shape)

    def prepare_gender_data(self) -> None:
        keep_cols = [
            "id",
            "sex",
            "decile_score",
            "is_recid",
            "two_year_recid",
            "c_charge_degree",
            "priors_count",
        ]
        data = self.__dataset[keep_cols].copy()

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

        output_dir = self.output_dir / "gender"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        plotKDEPerGroup(
            data,
            {0: "Male", 1: "Female"},
            "pred_score",
            output_dir / "dataPlot_sex.png",
        )
        data.to_csv(output_dir / "data.csv", index=False, header=True)
        print(f"compas data grouped by gender output to '{output_dir}'")

    def prepare_race_data(self) -> None:
        keepCols = [
            "id",
            "race",
            "decile_score",
            "is_recid",
            "two_year_recid",
            "c_charge_degree",
            "priors_count",
        ]
        data = self.__dataset[keepCols].copy()

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

        output_dir = self.output_dir / "race"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        plotKDEPerGroup(
            data,
            {0: "Caucasian", 1: "Afr.-Amer.", 2: "Hispanic", 3: "Other"},
            "pred_score",
            output_dir / "dataPlot_race.png",
        )
        data.to_csv(output_dir / "data.csv", index=False, header=True)
        print(f"compas data grouped by race output to '{output_dir}'")

    def prepare_age_data(self) -> None:
        keepCols = [
            "id",
            "age_cat",
            "decile_score",
            "is_recid",
            "two_year_recid",
            "c_charge_degree",
            "priors_count",
        ]
        data = self.__dataset[keepCols].copy()

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

        output_dir = self.output_dir / "age"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        plotKDEPerGroup(
            data,
            {0: "$> 45$", 1: "$25 - 45$", 2: "$< 25$"},
            "pred_score",
            output_dir / "dataPlot_age.png",
        )
        data.to_csv(output_dir / "data.csv", index=False, header=True)
        print(f"compas data grouped by age output to '{output_dir}'")
