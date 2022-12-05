# FAIM

FAIM (FAir Interpolation Method), described in
[Beyond Incompatibility: Interpolation between Mutually
Exclusive Fairness Criteria in Classification Problems](https://arxiv.org/abs/2212.00469)
is a post-processing algorithm for achieving a combination of group-fairness criteria
(equalized false postive rates, equalized false negative rates, group calibration).

[this page is under contruction]

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

[A general API will added soon]

### Experiments
#### Datasets
The CLI can be used to prepare the three datasets found in the paper:
```bash
faim-experiment --create <dataset>
```
where `<dataset>` is one of:
1. synthetic
2. compas
3. zalando

(see [paper](https://arxiv.org/abs/2212.00469) for more information).

##### Synthetic Dataset
The synthetic dataset contains 2 protected feature columns and 1 score column.
One protected feature is binary {0, 1}, the other is drawn from the set {0, 1, 2},
leading to a total of 6 groups (0 0), (0 1), (0 2) etc. The score feature represents the number,
that would be calculated by a ranking function. Each group is assigned a in integer score within [1,100],
drawn from a normal distribution with different means and standard-deviations per group.

#### Run Experiment

For each dataset the aforementioned group description csv file is needed. It is automatically generated during ``python3 main.py --create.``

Running the CFA requires the following parameters: dataset name, the lowest and highest score value, the step size between two consecutive score values, a theta value for each group, and a path where the results are stored

Examples for the synthetic dataset:
* ``continuous-kleinberg --run synthetic 1,100 1 0,0,0,0,0,0 ../data/synthetic/results/theta=0/``
* ``continuous-kleinberg --run synthetic 1,100 1 1,1,1,1,1,1 ../data/synthetic/results/theta=1/``

Example for LSAT with gender as protected feature:
* ``continuous-kleinberg --run lsat_gender 11,48 1 0,0 ../data/LSAT/gender/results/theta=0/``

Example for LSAT with race as protected feature:
* ``continuous-kleinberg --run lsat_race 11,48 1 1,1,1,1,1,1,1,1 ../data/LSAT/allRace/results/theta=1/``


#### Visualize Data and Results
Evaluates relevance and fairness changes for a given experiment and plots the results. Relevance is evaluated in terms of NDCG and Precision@k. Fairness is evaluated in terms of percentage of protected candidates at position k.

Running the evaluation requires the following terminal arguments: dataset name, path to original dataset (before post-processing with CFA), path to result dataset (after applying the CFA). The evaluation files are stored in the same directory as the result dataset.

* ``continuous-kleinberg --evaluate synthetic ../data/synthetic/dataset.csv ../data/synthetic/results/theta=0/resultData.csv``
* ``continuous-kleinberg --evaluate lsat_race ../data/LSAT/allRace/allEthnicityLSAT.csv ../data/LSAT/allRace/results/theta=0/resultData.csv``
* ``continuous-kleinberg --evaluate lsat_gender ../data/LSAT/gender/genderLSAT.csv ../data/LSAT/gender/results/theta=0/resultData.csv``


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
