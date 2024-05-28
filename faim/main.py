import argparse
import glob
import os
from datetime import datetime
from pathlib import Path
from time import process_time
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import pooch
from numpy._typing import NDArray

from faim import evaluation
from faim.algorithm.faim import FairInterpolationMethod
from faim.data_preparation.compas import CompasCreator
from faim.data_preparation.synthetic import SyntheticDatasetCreator
from faim.visualization.plots import plotScoreKDEsPerGroup

DATA_TOP_DIR = Path(__file__).parent.parent / "data"

# ToDo: Add CLI parameter for output dir instead of hard-coding
OUTPUT_DIR = Path(".") / "prepared-data"


def create_synthetic_data(size: int, group_names: Dict[int, str]) -> None:
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


def download_synthetic_data(
    base_url: str = "https://raw.githubusercontent.com/MilkaLichtblau/faim/main/data/synthetic/2groups/2022-01-12",
) -> None:
    SYNTHETIC_DATASET_OUTPUT_DIR = OUTPUT_DIR / "/".join(base_url.split("/")[-3:])
    if not SYNTHETIC_DATASET_OUTPUT_DIR.exists():
        SYNTHETIC_DATASET_OUTPUT_DIR.mkdir(parents=True)

    for filename, known_hash in zip(
        ("dataset.csv", "trueAndPredictedScoreDistributionPerGroup.png"),
        (
            "9637ae334d7f0d66b224f7e6f5c231d184efea6ce543d1814426b71541e815c8",
            "4a57a4f52d8763147abe4d00f876560ccbf9b4adb9fbc380aa0cbb80b4e7090b",
        ),
    ):
        if base_url[-1] == "/":
            base_url = base_url[:-1]

        pooch.retrieve(
            url=f"{base_url}/{filename}", known_hash=known_hash, path=SYNTHETIC_DATASET_OUTPUT_DIR, fname=filename
        )


def interpolate_fairly(
    score_stepsize: float,
    thetas: Dict[int, NDArray[np.float64]],
    result_dir: Path,
    data_filepath: Path,
    pred_score_column: str,
    group_names: Dict[int, str],
    optimal_transport_regularization: float,
):
    data = pd.read_csv(data_filepath, sep=",")

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
    # check that group thetas are not all zero
    for groupThetas in thetas.values():
        if all(t == 0 for t in groupThetas):
            raise ValueError("group thetas are all 0")

    t = process_time()
    fair_interpolation_method = FairInterpolationMethod(
        rawData=data,
        group_names=group_names,
        pred_score_column=pred_score_column,
        score_stepsize=score_stepsize,
        thetas=thetas,
        regForOT=optimal_transport_regularization,
        path=result_dir,
        plot=True,
    )
    result = fair_interpolation_method.run()
    elapsed_time = process_time() - t
    result.to_csv(result_dir / "resultData.csv")

    with (result_dir / "runtime.txt").open("a") as f:
        print(
            "running time: " + str(elapsed_time),
            file=f,
        )


def parse_thetas_arg(thetas_arg: str) -> NDArray[np.float64]:
    """Convert argv string of thetas into 2D array of floats of thetas, one row per group."""
    thetas_array = np.fromstring(thetas_arg, dtype=float, sep=",")
    # make sure three thetas exist for each group
    assert len(thetas_array) % 3 == 0
    return np.reshape(thetas_array, (-1, 3))


def main(argv: Optional[List[str]] = None):
    # parse command line options
    parser = argparse.ArgumentParser(epilog="=== === === end === === ===")

    parser.add_argument(
        "--prepare-data",
        nargs=1,
        choices=["synthetic-generated", "synthetic-from-paper", "compas", "zalando"],
        help="download (or generate new) raw data and prepare dataset for experiments (written to prepared-data folder in current directory)",
    )
    parser.add_argument(
        "--run",
        nargs=4,
        metavar=("DATASET NAME", "STEPSIZE", "THETAS", "PATH_TO_DATASET"),
        help="runs continuous fairness algorithm for given DATASET NAME with \
                              STEPSIZE and THETAS and stores results under a relative results directory",
    )
    parser.add_argument(
        "--evaluate",
        nargs=1,
        choices=["synthetic-generated", "synthetic-from-paper", "compas", "zalando"],
        help="evaluates all experiments for respective dataset and \
                              stores results into same directory",
    )

    args = parser.parse_args(argv)

    if args.prepare_data == ["synthetic-from-paper"]:
        download_synthetic_data(
            base_url="https://raw.githubusercontent.com/MilkaLichtblau/faim/main/data/synthetic/2groups/2022-01-12"
        )
    elif args.prepare_data == ["synthetic-generated"]:
        create_synthetic_data(size=100000, group_names={0: "privileged", 1: "disadvantaged"})
    elif args.prepare_data == ["compas"]:
        compasPreps = CompasCreator(output_dir=OUTPUT_DIR / "compas")
        compasPreps.prepare_gender_data()
        compasPreps.prepare_race_data()
        compasPreps.prepare_age_data()
    elif args.prepare_data == ["zalando"]:
        raise ValueError("The Zalando dataset has not yet been released. Please contact the authors for more info.")

        # ZalandoDataset(input_filepath=input_filepath, output_dir=OUTPUT_DIR / "zalando")
    elif args.run:
        score_stepsize = float(args.run[1])
        # FIXME: thetas are given as np matrix in same order of group names that are defined below, because I did not find a way to pass them as
        # dict
        thetas_array: NDArray[np.float64] = parse_thetas_arg(args.run[2])

        # create result directory with matching subdir structure as in data folder, assuming relative path
        data_filepath = Path(args.run[3])
        if not data_filepath.exists():
            raise FileNotFoundError(f"Data file not found: {data_filepath}")

        results_dir = Path(f"results/{'/'.join(data_filepath.parts[-4:-1])}/{args.run[2]}")
        results_dir.mkdir(parents=True, exist_ok=True)

        group_names: Dict[int, str]
        if args.run[0] == "synthetic-from-paper":
            optimal_transport_regularization = 0.001
            group_names = {0: "advantaged", 1: "disadvantaged"}
        elif args.run[0] == "compasGender":
            optimal_transport_regularization = 0.005
            group_names = {0: "male", 1: "female"}
        elif args.run[0] == "compasRace":
            optimal_transport_regularization = 0.05
            group_names = {0: "Caucasian", 1: "Afr.-Amer.", 2: "Hispanic", 3: "Other"}
        elif args.run[0] == "compasAge":
            optimal_transport_regularization = 0.005
            group_names = {0: "old", 1: "mid-age", 2: "young"}
        elif args.run[0] == "zalando":
            optimal_transport_regularization = 0.001
            group_names = {0: "low", 1: "high"}
        else:
            parser.error(
                "unknown dataset. Options are 'synthetic-from-paper', \
                'compasRace', 'compasAge', 'compasGender', and \
                'zalando'."
            )
            assert False

        thetas = {group_id: thetas_array[idx] for idx, group_id in enumerate(group_names.keys())}

        interpolate_fairly(
            score_stepsize=score_stepsize,
            thetas=thetas,
            result_dir=results_dir,
            data_filepath=data_filepath,
            pred_score_column="pred_score",
            group_names=group_names,
            optimal_transport_regularization=optimal_transport_regularization,
        )
    elif args.evaluate:
        if args.evaluate[0] == "synthetic-from-paper":
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
