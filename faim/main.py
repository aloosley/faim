import argparse
import glob
import os
from datetime import datetime
from pathlib import Path
from time import process_time
from typing import Dict

import numpy as np
import pandas as pd

from faim import evaluation
from faim.algorithm.fia import FairnessInterpolationAlgorithm
from faim.data_preparation.compas import CompasCreator
from faim.data_preparation.synthetic import SyntheticDatasetCreator
from faim.data_preparation.zalando import ZalandoDataset
from faim.visualization.plots import plotScoreKDEsPerGroup


DATA_TOP_DIR = Path(__file__).parent.parent / "data"

# ToDo: Add CLI parameter for output dir instead of hard-coding
OUTPUT_DIR = Path(".") / "prepared-data"


def create_synthetic_data(size: int, group_names: Dict[int, str]):
    creator = SyntheticDatasetCreator(size, len(group_names))
    creator.createTwoCorrelatedNormalDistributionScores()
    creator.sortByColumn("pred_score")
    creator.setDecisionBoundaryAsMean("true_score", "pred_score")

    # create subdir structure
    groupStr = str(len(group_names)) + "groups/"
    timestampStr = datetime.today().strftime("%Y-%m-%d")

    output_dir = OUTPUT_DIR / "synthetic" / groupStr / timestampStr
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # store
    creator.writeToCSV(output_dir / "dataset.csv")
    plotScoreKDEsPerGroup(
        creator.dataset,
        np.arange(len(group_names)),
        ["true_score", "pred_score"],
        creator.boundary,
        output_dir / "trueAndPredictedScoreDistributionPerGroup.png",
        group_names,
    )
    print(f"synthetic data output to '{output_dir}'")


def interpolate_fairly(score_stepsize, thetas, result_dir, pathToData, pred_score, group_names, regForOT):
    data = pd.read_csv(pathToData, sep=",")

    # check that we have a theta for each group and for each fairness criterion
    actual = 0
    for v in thetas.values():
        actual += len(v)
    if (3 * len(group_names.keys())) != actual:
        raise ValueError(
            "invalid number of thetas, should be {expected}, but was {actual}. \
                         Specify three thetas per group.".format(
                expected=3 * len(group_names.keys()), actual=actual
            )
        )

    t = process_time()
    fia = FairnessInterpolationAlgorithm(
        data,
        group_names,
        pred_score,
        score_stepsize,
        thetas,
        regForOT,
        path=result_dir,
        plot=True,
    )
    result = fia.run()
    elapsed_time = process_time() - t
    result.to_csv(os.path.join(result_dir, "resultData.csv"))

    print(
        "running time: " + str(elapsed_time),
        file=open(os.path.join(result_dir, "runtime.txt"), "a"),
    )


def parseThetasToMatrix(thetaString):
    """Convert argv string of thetas into 2D array of floats of thetas, one row per group."""
    thetas1D = np.fromstring(thetaString, dtype=float, sep=",")
    # make sure three thetas exist for each group
    assert len(thetas1D) % 3 == 0
    return np.reshape(thetas1D, (-1, 3))


def main():
    # parse command line options
    parser = argparse.ArgumentParser(prog="Continuous Kleinberg Interpolation", epilog="=== === === end === === ===")

    parser.add_argument(
        "--create",
        nargs=1,
        choices=["synthetic", "compas", "zalando"],
        help="creates datasets from raw data and writes them to disk",
    )
    parser.add_argument(
        "--run",
        nargs=4,
        metavar=("DATASET NAME", "STEPSIZE", "THETAS", "DIRECTORY"),
        help="runs continuous fairness algorithm for given DATASET NAME with \
                              STEPSIZE and THETAS and stores results into DIRECTORY",
    )
    parser.add_argument(
        "--evaluate",
        nargs=1,
        choices=["synthetic", "compas", "zalando"],
        help="evaluates all experiments for respective dataset and \
                              stores results into same directory",
    )

    args = parser.parse_args()

    if args.create == ["synthetic"]:
        create_synthetic_data(100000, {0: "privileged", 1: "disadvantaged"})
    elif args.create == ["compas"]:
        compasPreps = CompasCreator(
            data_filepath=DATA_TOP_DIR / "compas" / "compas_two_years.csv", output_dir=OUTPUT_DIR / "compas"
        )
        compasPreps.prepare_gender_data()
        compasPreps.prepare_race_data()
        compasPreps.prepare_age_data()
    elif args.create == ["zalando"]:
        input_filepath = DATA_TOP_DIR / "zalando/raw-data.csv"

        if not input_filepath.exists():
            raise FileNotFoundError("The Zalando dataset must be requested from the authors.")

        ZalandoDataset(input_filepath=input_filepath, output_dir=OUTPUT_DIR / "zalando")
    elif args.run:
        score_stepsize = float(args.run[1])
        # FIXME: thetas are given as np matrix in same order of group names that are defined below, because I did not find a way to pass them as
        # dict
        thetasAsNpMatrix = parseThetasToMatrix(args.run[2])

        # set current working directory to /Code such that paths are not messed up on different systems
        os.chdir(Path(__file__).resolve().parent.parent.absolute())
        # create result directory with matching subdir structure as in data folder, assuming relative path
        relativePathToData = args.run[3]
        # extract subdir structure
        folderList = os.path.normpath(relativePathToData).split(os.path.sep)
        # replace top-level dir and delete filename
        folderList = folderList[1:-1]
        folderList.insert(0, "results")
        # add folder for thetas
        folderList.append(args.run[2])
        resultDir = os.path.join(*(folderList))
        Path(resultDir).mkdir(parents=True, exist_ok=True)

        if args.run[0] == "syntheticTwoGroups":
            regForOT = 0.001
            groupNames = {0: "privileged", 1: "disadvantaged"}
        elif args.run[0] == "compasGender":
            regForOT = 0.005
            groupNames = {0: "male", 1: "female"}
        elif args.run[0] == "compasRace":
            regForOT = 0.05
            groupNames = {0: "Caucasian", 1: "Afr.-Amer.", 2: "Hispanic", 3: "Other"}
        elif args.run[0] == "compasAge":
            regForOT = 0.005
            groupNames = {0: "old", 1: "mid-age", 2: "young"}
        elif args.run[0] == "zalando":
            regForOT = 0.001
            groupNames = {0: "low", 1: "high"}
        else:
            parser.error(
                "unknown dataset. Options are 'syntheticTwoGroups', \
                'compas race', 'compas age', 'compas gender', and \
                'zalando'."
            )

        thetas = dict(
            (groupName, thetasAsNpMatrix[i]) for groupName in groupNames.keys() for i in range(len(groupNames.keys()))
        )
        interpolate_fairly(
            score_stepsize,
            thetas,
            resultDir,
            relativePathToData,
            "pred_score",
            groupNames,
            regForOT,
        )
    elif args.evaluate:
        if args.evaluate[0] == "synthetic":
            allSyntheticResults = glob.glob(
                os.path.join("results", "synthetic", "**", "resultData.csv"),
                recursive=True,
            )
            for pathToResult in allSyntheticResults:
                resultData = pd.read_csv(pathToResult, sep=",")
                evaluation.synthetic.evaluate(resultData, pathToResult)
        if args.evaluate[0] == "compas":
            allCompasResults = glob.glob(
                os.path.join("results", "compas", "**", "resultData.csv"),
                recursive=True,
            )
            for pathToResult in allCompasResults:
                resultData = pd.read_csv(pathToResult, sep=",")
                evaluation.compas.evaluate(resultData, pathToResult)
                evaluation.compas.evaluatePerBin(resultData, pathToResult)
        if args.evaluate[0] == "zalando":
            allZalandoResults = glob.glob(
                os.path.join("results", "zalando", "**", "resultData.csv"),
                recursive=True,
            )
            for pathToResult in allZalandoResults:
                resultData = pd.read_csv(pathToResult, sep=",")
                evaluation.zalando.evaluate(resultData, pathToResult)
    else:
        parser.error("choose one command line option")


if __name__ == "__main__":
    main()
