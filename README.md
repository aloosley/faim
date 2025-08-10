[![Build & Test](https://github.com/MilkaLichtblau/faim/actions/workflows/python-build-test.yaml/badge.svg)](https://github.com/MilkaLichtblau/faim/actions/workflows/python-build-test.yaml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![PyPI](https://img.shields.io/pypi/v/faim?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/faim)

# FAIM

FAIM (FAir Interpolation Method)
is a post-processing algorithm for achieving a combination of group-fairness criteria
(equalized false positive rates, equalized false negative rates, group calibration).

## Citing
This work was [published in 2025](https://www.sciencedirect.com/science/article/pii/S0004370224002169) in the Journal of Artificial Intelligence.  If you use FAIM or use ideas from this work to develop your own algorithms, please cite the paper:

```bibtex
@article{ZEHLIKE2025104280,
  title = {Beyond incompatibility: Trade-offs between mutually exclusive fairness criteria in machine learning and law},
  journal = {Artificial Intelligence},
  volume = {340},
  pages = {104280},
  year = {2025},
  issn = {0004-3702},
  doi = {https://doi.org/10.1016/j.artint.2024.104280},
  url = {https://www.sciencedirect.com/science/article/pii/S0004370224002169},
  author = {Meike Zehlike and Alex Loosley and Håkan Jonsson and Emil Wiedemann and Philipp Hacker}
}
```

🚧 **This README.md is under construction!** 🚧

## Getting Started

### Environment
Ensure you have a environment with Python>=3.9, preferably by creating a virtual environment.

One way to do this is using [miniconda](https://docs.conda.io/en/latest/miniconda.html).  Install miniconda following
the instructions on [this page](https://docs.conda.io/en/latest/miniconda.html)
and create a python 3.10 environment:

```bash
conda create -n faim python=3.12
```

Activate the environment
```bash
conda activate faim
```

Check that the version of python running comes from the faim environment:
```bash
which python
```

### Installation
There are two implementations of FAIM in the FAIM package, one is the origianl research code
which can be used to reproduce all experimental results and figures from the paper.  The other is
a revamped implementation designed to be merged into [Fairlearn](fairlearn.org).  This section covers both.

Note, if you intend to develop the package and/or contribute, follow the install instructions in the
[Development Environment](#development-environment) section below instead.  Otherwise, follow these instructions.


#### Fairlearn Implementation
As of 16.07.2024, this is a work in progress that will stabilize once the code is merged with Fairlearn,
in which case this section will be obsolete.  But to use the current version of FAIM to post-process
your scores to fairer scores, simply install FAIM as follows and scroll down
to the [Usage](#usage) section for instructions on how to use the FAIM Fairlearn API.

```commandline
pip install faim
```


#### Paper Implementation

The package and experiment CLI used to reproduce all the results in the [paper](https://arxiv.org/abs/2212.00469)
can be installed with:
```bash
pip install "faim[experiment]"
```

Many of the figures are rendered with LaTeX (via Matplotlib) and require latex be installed.

See [this Matplotlib documentation page](https://matplotlib.org/stable/users/explain/text/usetex.html#text-rendering-with-latex) for instructions.

If you're on a Mac, you can install the LaTeX distribution MacTeX using [brew cask](https://formulae.brew.sh/cask/):
```bash
brew install --cask mactex
```

### Removal
From the environment where you installed the package, run
```bash
pip uninstall faim
```


## Usage
### Fairlearn Implementation
> ⚠️ **WARNING** ⚠️
>
> The goal is to merge the code into the [Fairlearn](https://fairlearn.org/) library making it available under
the post-processing submodule. This version of the code is being prepared for a pull request into the
[Fairlearn](https://fairlearn.org/) and should be considered **unstable** and subject to change as we
prepare and test this new implementation. Use with caution! If stability is important and you do not want to wait
to use FAIM, consider using the [Paper Implementation](#paper-implementation) for now.

In FAIM, the user chooses via hyperparameter `theta` how to balance between otherwise
mutually exclusive fairness criteria.

FAIM currently supports three fairness criteria:

1. Calibration between groups (scores actually correspond to probability of positive)
1. Balance for the negative class (average score of truly negative individuals equal across groups)
1. Balance for the positive class (average score of truly positive individuals equal across groups)

The sections below show a usage example using the Fairlearn implementation of FAIM, which
can also be run via [notebooks/fairlearn-api.ipynb](notebooks/faim-scores-example.ipynb) .


#### Data
Load some test data.
```python
import pandas as pd

synthetic_data_from_paper = pd.read_csv(
    "https://raw.githubusercontent.com/MilkaLichtblau/faim/main/data/synthetic/2groups/2022-01-12/dataset.csv"
)

y_scores = synthetic_data_from_paper.pred_score
y_scores = (y_scores - y_scores.min())/(y_scores.max() - y_scores.min())
y_ground_truth = synthetic_data_from_paper.groundTruthLabel.astype(bool)
sensitive_features = synthetic_data_from_paper.group
```
Note above, the scores above are normalized between 0 and 1 because FAIM expects this to be able calculate
meaningful fair score distributions (don't worry, FAIM will raise an error if non normalized scores are passed).

#### Fit
Now fit a FAIM model that maps scores between a balance of one or more of the three fairness criteria above (below
a balance between calibration and balance for the negative class is used for example purposes):

```python
from faim.fairlearn_api import FAIM

thetas = [1/2, 1/2, 0]  # balance between calibration and balance for the negative class
faim = FAIM(thetas=thetas)
faim.fit(y_scores, y_ground_truth, sensitive_features)
```

#### Predict
To map new scores with the fitted faim model, use the predict method:
```python
faim.predict(y_scores, sensitive_features)
```


### Paper Implementation
This section contains information for reproducing experiments in our [paper](https://arxiv.org/abs/2212.00469).

Ensure the package has been installed with `[experiment]` extra requirements before continuing
(see [Installation | Paper Implementation](#python-package))!  Don't forget to restart your terminal before
using the `faim` CLI in the steps below.

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

For any folder with results, the following command can be used to evaluate performance and error rates before and after
the application of FAIM:
```bash
faim-experiment --evaluate DATASET
```
where `DATASET` is one of the following options:
* `synthetic-from-paper` (recursively searches for all resultData.csv under results/synthetic)
* `compas` (recursively searches for all resultData.csv under results/compas)
* `zalando` (recursively searches for all resultData.csv under results/zalando)

For example, if you run `faim-experiment --evaluate synthetic-from-paper`,
then for any (relative) experiment folder under `results/synthetic/` that contains the file `resultsData.csv`,
an `eval.txt` file will be created containing following metrics:
* The probability of the protected groups to be labeled positive w.r.t. the non-protected group, for the three cases ground truth, original prediction, and fair score prediction.
* Accuracy, Precision, and Recall (by class and macro/weighted averages) for the original and the fair model, plus the difference between them
* False positive and false negative rates for the original and the fair model, plus the differcence between them
* Confusion matrices for each group and all groups

Should you wish to calculate other metrics beyond what are shown in the paper, `eval.txt` should provide everything needed to do so.

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
