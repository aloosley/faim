#!/bin/sh
# runs experiments on synthetic dataset, COMPAS dataset, and Zalando dataset with different thetas

# SYNTHETIC TWO GROUPS
echo python3 main.py --run syntheticTwoGroups 0.1 1,1,1,1,1,1 data/synthetic/2groups/2022-01-12/dataset.csv
python3 main.py --run syntheticTwoGroups 0.1 1,1,1,1,1,1 data/synthetic/2groups/2022-01-12/dataset.csv

echo python3 main.py --run syntheticTwoGroups 0.1 1,0,0,1,0,0 data/synthetic/2groups/2022-01-12/dataset.csv
python3 main.py --run syntheticTwoGroups 0.1 1,0,0,1,0,0 data/synthetic/2groups/2022-01-12/dataset.csv

echo python3 main.py --run syntheticTwoGroups 0.1 0,1,0,0,1,0 data/synthetic/2groups/2022-01-12/dataset.csv
python3 main.py --run syntheticTwoGroups 0.1 0,1,0,0,1,0 data/synthetic/2groups/2022-01-12/dataset.csv

echo python3 main.py --run syntheticTwoGroups 0.1 0,0,1,0,0,1 data/synthetic/2groups/2022-01-12/dataset.csv
python3 main.py --run syntheticTwoGroups 0.1 0,0,1,0,0,1 data/synthetic/2groups/2022-01-12/dataset.csv

# COMPAS GENDER
echo python3 main.py --run compasGender 1 1,1,1,1,1,1 data/compas/gender/data.csv
python3 main.py --run compasGender 1 1,1,1,1,1,1 data/compas/gender/data.csv

echo python3 main.py --run compasGender 1 1,0,0,1,0,0 data/compas/gender/data.csv
python3 main.py --run compasGender 1 1,0,0,1,0,0 data/compas/gender/data.csv

echo python3 main.py --run compasGender 1 0,1,0,0,1,0 data/compas/gender/data.csv
python3 main.py --run compasGender 1 0,1,0,0,1,0 data/compas/gender/data.csv

echo python3 main.py --run compasGender 1 0,0,1,0,0,1 data/compas/gender/data.csv
python3 main.py --run compasGender 1 0,0,1,0,0,1 data/compas/gender/data.csv

# COMPAS RACE
echo python3 main.py --run compasRace 1 1,1,1,1,1,1,1,1,1,1,1,1 data/compas/race/data.csv
python3 main.py --run compasRace 1 1,1,1,1,1,1,1,1,1,1,1,1 data/compas/race/data.csv

echo python3 main.py --run compasRace 1 1,0,0,1,0,0,1,0,0,1,0,0 data/compas/race/data.csv
python3 main.py --run compasRace 1 1,0,0,1,0,0,1,0,0,1,0,0 data/compas/race/data.csv

echo python3 main.py --run compasRace 1 0,1,0,0,1,0,0,1,0,0,1,0 data/compas/race/data.csv
python3 main.py --run compasRace 1 0,1,0,0,1,0,0,1,0,0,1,0 data/compas/race/data.csv

echo python3 main.py --run compasRace 1 0,0,1,0,0,1,0,0,1,0,0,1 data/compas/race/data.csv
python3 main.py --run compasRace 1 0,0,1,0,0,1,0,0,1,0,0,1 data/compas/race/data.csv

# COMPAS AGE
echo python3 main.py --run compasAge 1 1,1,1,1,1,1,1,1,1 data/compas/age/data.csv
python3 main.py --run compasAge 1 1,1,1,1,1,1,1,1,1 data/compas/age/data.csv

echo python3 main.py --run compasAge 1 1,0,0,1,0,0,1,0,0 data/compas/age/data.csv
python3 main.py --run compasAge 1 1,0,0,1,0,0,1,0,0 data/compas/age/data.csv

echo python3 main.py --run compasAge 1 0,1,0,0,1,0,0,1,0 data/compas/age/data.csv
python3 main.py --run compasAge 1 0,1,0,0,1,0,0,1,0 data/compas/age/data.csv

echo python3 main.py --run compasAge 1 0,0,1,0,0,1,0,0,1 data/compas/age/data.csv
python3 main.py --run compasAge 1 0,0,1,0,0,1,0,0,1 data/compas/age/data.csv

# ZALANDO LOW VS HIGH VISIBILITY BRANDS
echo python3 main.py --run zalando 0.01 1,1,1,1,1,1 data/zalando/data.csv
python3 main.py --run zalando 0.01 1,1,1,1,1,1 data/zalando/data.csv

echo python3 main.py --run zalando 0.01 1,0,0,1,0,0 data/zalando/data.csv
python3 main.py --run zalando 0.01 1,0,0,1,0,0 data/zalando/data.csv

echo python3 main.py --run zalando 0.01 0,1,0,0,1,0 data/zalando/data.csv
python3 main.py --run zalando 0.01 0,1,0,0,1,0 data/zalando/data.csv

echo python3 main.py --run zalando 0.01 0,0,1,0,0,1 data/zalando/data.csv
python3 main.py --run zalando 0.01 0,0,1,0,0,1 data/zalando/data.csv
