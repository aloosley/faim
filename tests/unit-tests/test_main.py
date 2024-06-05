import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from faim.main import main


def test_main_prepare_synthetic_data() -> None:
    # GIVEN temporary location for stored data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        with patch("faim.main.OUTPUT_DIR", temp_dir):
            expected_prepared_data_directory = Path(
                f"{temp_dir}/synthetic/2groups/{datetime.today().strftime('%Y-%m-%d')}"
            )

            # WHEN prepare-data invoked via CLI
            main(["--prepare-data", "synthetic-generated"])

            # THEN data is as expected
            dict(pd.read_csv(expected_prepared_data_directory / "dataset.csv").iloc[0]) == {
                "groundTruthLabel": 1,
                "group": 0,
                "pred_score": 6.77016283601609,
                "predictedLabel": 1,
                "true_score": 5.491779193811632,
                "uuid": "268234863050756571371918738194881370874",
            }


@pytest.mark.optional
def test_regression_test_running_faim_on_synthetic_data_from_paper() -> None:
    # GIVEN temporary location for stored data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        # temp_dir = Path("./temp-dir")

        with patch("faim.main.OUTPUT_DIR", temp_dir):
            expected_prepared_data_directory = Path(f"{temp_dir}/synthetic/2groups/2022-01-12")

            # GIVEN synthetic data prepared
            main(["--prepare-data", "synthetic-from-paper"])

            # WHEN running experiment on synthetic data
            main(
                [
                    "--run",
                    "synthetic-from-paper",
                    "0.1",
                    "1,0,0,1,0,0",
                    str(expected_prepared_data_directory / "dataset.csv"),
                ]
            )

            main(["--evaluate", "synthetic-from-paper"])

            eval_results_filepath = Path("./results/synthetic/2groups/2022-01-12/1,0,0,1,0,0/eval.txt")
            with eval_results_filepath.open("r") as f:
                eval_results = f.readlines()

            assert [float(val) for val in eval_results[-12].split()[1:]] == [
                0.852,
                0.853,
                0.852,
                0.885,
                0.885,
                0.885,
                -0.033,
                -0.032,
                -0.033,
            ]
            assert [float(val) for val in eval_results[-11].split()] == [
                0.0,
                0.86,
                0.868,
                0.86,
                0.884,
                0.873,
                0.884,
                -0.024,
                -0.005,
                -0.024,
            ]
            assert [float(val) for val in eval_results[-10].split()] == [
                1.0,
                0.844,
                0.869,
                0.844,
                0.885,
                0.874,
                0.885,
                -0.041,
                -0.005,
                -0.041,
            ]
