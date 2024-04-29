# ads_case_problem

## Overview

This is a Kedro project, which was generated using `Kedro 0.17.7`.

## Test results 

The test results are in the file : `data/07_model_output/test.csv` 

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

The data paths are stored in the Data catalog in the file : `conf/base/catalog.yml`

The project parameters are in the file : `conf/base/parameters.yml`

## Model selection and data analysis

The model selection and data analysis are in the file : `analysis.ipynb`

## Pipelines

The project consists of different pipelines in the file : `src/ads/pipeline_registry.py` :

The user_app pipeline allows predicting : 

```
kedro run --pipeline user_app
```

The output will be saved in this path: `data/07_model_output/test.csv`

To use other pipelines : 

```
kedro run --pipeline pipeline_name
```
