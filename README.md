[![Build & Test](https://github.com/MilkaLichtblau/faim/actions/workflows/python-build-test.yaml/badge.svg)](https://github.com/MilkaLichtblau/faim/actions/workflows/python-build-test.yaml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# FAIM

FAIM (FAir Interpolation Method), described in
[Beyond Incompatibility: Interpolation between Mutually
Exclusive Fairness Criteria in Classification Problems](https://arxiv.org/abs/2212.00469),
is a post-processing algorithm for achieving a combination of group-fairness criteria
(equalized false positive rates, equalized false negative rates, group calibration).

**This README.md is under construction!**

## Installation

### Environment
Ensure you have a environment with Python>=3.8 and pip>=2.21, preferably by creating a virtual environment.

One way to do this is using [miniconda](https://docs.conda.io/en/latest/miniconda.html).  Install miniconda following
the instructions on [this page](https://docs.conda.io/en/latest/miniconda.html)
and create a python 3.10 environment:

```bash
conda create -n faim python=3.10
```

Activate the environment
```bash
conda activate faim
```

Check that versions of python are >=3.8 and >=2.21, respectively:
```bash
python --version
pip --version
```

### Python Package
If you intend to develop the package and/or contribute, follow the install instructions in the
[Development Environment](#development-environment) section below instead.  Otherwise, follow these instructions.

The package and experiment CLI can be installed with pip:
```bash
pip install "faim[experiment]"
```

Note the `[experiment]` notation is required for now since, for the moment, the algorithm can only be run in experiment
mode for recreating experimental results in the [paper](https://arxiv.org/abs/2212.00469).
In the future, `faim` will be made available for post-processing classifier scores
(given ground truth and group information), going beyond reproducing paper experiments.



### Removal
From the environment where you installed the package, run
```bash
pip uninstall faim
```

### Latex
Many of the figures are rendered with LaTeX (via Matplotlib) and require latex be installed.

See [this Matplotlib documentation page](https://matplotlib.org/stable/users/explain/text/usetex.html#text-rendering-with-latex) for instructions.

If you're on a Mac, you can install the LaTeX distribution MacTeX using [brew cask](https://formulae.brew.sh/cask/):
```bash
brew install --cask mactex
```
Don't forget to restart your terminal before using the `faim` CLI.

## Usage
See [notebooks/faim-scores-example.ipynb](notebooks/faim-scores-example.ipynb) for an example of
calculating faim scores.

### Experiments
This section contains information for reproducing experiments in our [paper](https://arxiv.org/abs/2212.00469).

Ensure the package has been installed with `[experiment]` extra requirements before continuing
(see [Installation | Python Package](#python-package))!

#### Prepare Data
The CLI can be used to prepare any of the three datasets used in the [paper](https://arxiv.org/abs/2212.00469):
```bash
faim-experiment --prepare-data DATASET
```
where `DATASET` is one of:
* `synthetic-from-paper`
* `compas`
* `zalando` [waiting for permission to release, contact us for more information]

The dataset will be downloaded, and prepared to a folder called `prepared-data`.

The following sections include info about each dataset:

###### Synthetic data
The raw dataset in the GitHub repo corresponds to synthetic prediction and ground truth scores for two groups,
for each group sampling from a corresponding binormal distribution.

###### COMPAS data
The raw data was obtained from [ProPublica's COMPAS Analysis repository](https://github.com/propublica/compas-analysis).

###### Zalando data
**Under construction, more information to follow!**

#### Run Experiment

Having prepared data following the instruction above, you are ready to run a FAIM experiment:
```bash
faim-experiment --run PREPARED-DATASET LOW_SCORE_VAL,HIGH_SCORE_VAL THETAS PREPARED_DATA_FILEPATH
```

`PREPARED-DATASET` is now one of the following options (depending on what has been prepared):
* `synthetic-from-paper` (prepared using `--prepare-data synthetic-from-paper`)
* `compasGender` (prepared using `--prepare-data compas`)
* `compasRace` (prepared using `--prepare-data compas`)
* `compasAge` (prepared using `--prepare-data compas`)
* `zalando` (prepared using `--prepare-data zalando`) [waiting for permission to release, contact us for more information]

`LOW_SCORE_VAL,HIGH_SCORE_VAL` are two numbers that define the score range.

`THETAS` correspond to the fairness compromise you want. There are three thetas per group corresponding to the
desired amount of the three fairness criteria that the system should achieve:
1. group calibration
2. equalized false negative rates
3. equalized false positive rates

Note, as discussed in the paper, thetas = 1,1,1 does not indicate that the system will simultaneously achieve all
three (mutually incompatible) fairness criteria, but rather the result will be a compromise between all three.

See the [paper](https://arxiv.org/abs/2212.00469) for more details.

Finally, `PREPARED_DATA_FILEPATH` corresponds to the filepath of the prepared data.

###### Examples
Run all of the following from the same folder where `faim-experiment --prepare-data` was run.

In each example, a FAIM post-processor is trained and evaluated with results saved under the `results` folder:
* Train FAIM model on synthetic dataset with callibration as fairness correction
  ```bash
  faim-experiment --run synthetic-from-paper 0.1 1,0,0,1,0,0 prepared-data/synthetic/2groups/2022-01-12/dataset.csv
  ```
* Train FAIM model on synthetic dataset to achieve a combination of all three fairness criteria.
  ```bash
  faim-experiment --run synthetic-from-paper 0.1 1,1,1,1,1,1 prepared-data/synthetic/2groups/2022-01-12/dataset.csv
  ```

Note, that at the moment we do not allow all thetas to be 0.

#### Visualize and Evaluate Results
During the calculation the faim-algorithm creates a lot of plots that will help you to evaluate your results visually. You'll find them in the respective result folder of your experimental run, e.g., for experiment
  `faim-experiment --run synthetic-from-paper 0.1 1,0,0,1,0,0 prepared-data/synthetic/2groups/2022-01-12/dataset.csv`
all results are saved to `results/synthetic/2groups/2022-01-12/1,0,0,1,0,0/`.

These results include:
* resultData.csv, which contains the original dataset plus four a new columns: SA, SB, SC, and the final fair scores that correspond to the given thetas
* plot of the raw score distribution per group (truncatedRawScoreDistributionPerGroup.png)
* plots of SA, SB, and SC per group (muA_perGroup.png, SBDistributionPerGroup.png, SCDistributionPerGroup.png
* plot of the fair score distribution per group (fairScoreDistributionPerGroup.png)
* plots of the transport maps per group (fairScoreReplacementStrategy.png)

In addition to the plots, we provide performance and fairness evaluation code for all datasets. It automatically finds all resultData.csv-files for any experimental setup (e.g., different values of the theta-vector, or different datasets such as compasAge or compasGender) and writes a file eval.txt into the same folder of the respective resultData.csv.

To run the evaluation, type
```bash
faim-experiment --evaluate DATASETS
```
`DATASETS` is now one of the following options:
* `synthetic-from-paper` (recursively searches for all resultData.csv under results/synthetic)
* `compas` (recursively searches for all resultData.csv under results/compas)
* `zalando` (recursively searches for all resultData.csv under results/zalando)

eval.txt calculates the following metrics:
* The probability of the protected groups to be labeled positive w.r.t. the non-protected group, for the three cases ground truth, original prediction, and fair score prediction.
* Accuracy, Precision, and Recall for the original and the fair model, plus the difference between them
* False positive and false negative rates for the original and the fair model, plus the differcence between them

### Development Environment
To develop and/or contribute, clone the repository
```bash
git clone <this repo URL>
```

From the root directory of the git repository, install the package with pip in editable mode (`-e`)
with extra requirements for experiments (experiment) and development (dev):
```bash
pip install -e ".[experiment,dev]"
```

Don't confuse the `[]` to mean optional.  The `".[experiment, dev]"` notation tells pip to install extra
"experiment" and "dev" requirements including things like `pytest` and `pre-commit`.

When contributing, be sure to install (and use) our [pre-commit](https://pre-commit.com/) hooks:
```bash
pre-commit install -t pre-commit -t pre-push
```
