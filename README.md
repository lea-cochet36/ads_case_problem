# ads_case_problem

## Overview

This is your new Kedro project, which was generated using `Kedro 0.17.7`.

Take a look at the [Kedro documentation](https://kedro.readthedocs.io) to get started.

## Installation

To use this project, you must execute the following commands :

```
git clone https://github.com/lea-cochet36/ads_case_problem
cd ads_case_problem
```

You can also create a Conda environment (or equivalent) with the packages to install for the project:

```
conda env create -f environment.yml
conda activate ads_env
```

## Catalog et parameters

The data paths are stored in the Data catalog located in the file : `conf/base/catalog.yml`

The project parameters are located in the file : `conf/base/parameters.yml`

## Model selection and data analysis

The model selection and data analysis are located in the file : `analysis.ipynb`

## Pipelines

The project consists of different pipelines in the file : `src/ads/pipeline_registry.py` :

The user_app pipeline allows predicting on the test data. First, add the test file to the following path : `data/01_raw/test.csv` then execute the command : 

```
kedro run --pipeline user_app
```

The output will be saved in this path: `data/07_model_output/test_pred.pkl`

To use other pipelines, first check the data path in the Data Catalog, then execute the command :

```
kedro run --pipeline pipeline_name
```
