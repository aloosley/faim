[![Build & Test](https://github.com/MilkaLichtblau/faim/actions/workflows/python-build-test.yaml/badge.svg)](https://github.com/MilkaLichtblau/faim/actions/workflows/python-build-test.yaml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# FAIM

FAIM (FAir Interpolation Method), described in
[Beyond Incompatibility: Interpolation between Mutually
Exclusive Fairness Criteria in Classification Problems](https://arxiv.org/abs/2212.00469),
is a post-processing algorithm for achieving a combination of group-fairness criteria
(equalized false positive rates, equalized false negative rates, group calibration).

[this page is under construction]

## Installation

### Environment
Ensure you have a environment with Python>=3.7 and pip>=2.21, preferably by creating a virtual environment.

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

Check that versions of python are >=3.7 and >=2.21, respectively:
```bash
python --version
pip --version
```

### Python Package
To install the package, go to the root directory of this repository and run
```bash
pip install ".[experiment]"
```

Note the `[experiment]` notation is required for now since for the moment, the algorithm can only be run in experiment
mode for recreating experimental results in the [paper](https://arxiv.org/abs/2212.00469).
**In the future, `faim` will be made available directly via `pip install faim` with an API for easily applying the
post-processing algorithm to any classifier scores (given ground truth and group information).

### Removal
From the environment where you installed the package, run
```bash
pip uninstall faim
```

## Usage
Installing faim also (currently) installs one command line interface (CLI) tool, `faim-experiment` which can be
used to reproduce the work in the paper.

[A general API will be added soon]

### Experiments
This section contains information for reproducing experiments in our [paper](https://arxiv.org/abs/2212.00469).

Ensure the package has been installed with `[experiment]` extra requirements before continuing
(see [Installation | Python Package](#python-package))!

#### Prepare Data
The CLI can be used to prepare any of the three datasets used in the [paper](https://arxiv.org/abs/2212.00469):
```bash
faim-experiment --create DATASET
```
where `DATASET` is one of:
* `synthetic`
* `compas`
* `zalando` [waiting for permission to release, contact us for more information]

#### Run Experiment

For each dataset the aforementioned group description csv file is needed. It is automatically generated during ``python3 main.py --create.``

Running the CFA requires the following parameters: dataset name, the lowest and highest score value, the step size between two consecutive score values, a theta value for each group, and a path where the results are stored

Examples for the synthetic dataset:
* ``faim-experiment --run synthetic 1,100 1 0,0,0,0,0,0 ../data/synthetic/results/theta=0/``
* ``faim-experiment --run synthetic 1,100 1 1,1,1,1,1,1 ../data/synthetic/results/theta=1/``

#### Visualize Results
Evaluates relevance and fairness changes for a given experiment and plots the results. Relevance is evaluated in terms of NDCG and Precision@k. Fairness is evaluated in terms of percentage of protected candidates at position k.

Running the evaluation requires the following terminal arguments: dataset name, path to original dataset (before post-processing with CFA), path to result dataset (after applying the CFA). The evaluation files are stored in the same directory as the result dataset.

* ``faim-experiment --evaluate synthetic ../data/synthetic/dataset.csv ../data/synthetic/results/theta=0/resultData.csv``


## Development and Contribution
Contributions are welcome.

### Development Environment
To develop, use the following command to install the package as editable with extra dev requirements:
```bash
pip install -e ".[experiment, dev]"
```

Don't confuse the `[]` to mean optional.  The `".[experiment, dev]"` notation tells pip to install extra
"experiment" and "dev" requirements including things like `pytest`, `pre-commit`, `matplotlib`, and so on.

Please be sure to install (and use) our [pre-commit](https://pre-commit.com/) hooks:
```bash
pre-commit install -t pre-commit -t pre-push
```
