<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [MLProject - v0.0.0 (unreleased)](#mlproject---v000-unreleased)
  - [Clone](#clone)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Data](#data)
  - [Usage](#usage)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# MLProject - v0.0.0 (unreleased)

This repository contains a machine learning code to solve
the [Collision Avoidance Challenge](https://kelvins.esa.int/collision-avoidance-challenge/home/) from ESA released in
October 2019.

## Clone

* **HTTPS:** ```git clone https://github.com/freddida/MLProject.git```
* **SSH:** ```git clonegit@github.com:freddida/MLProject.git```

## Installation

### Requirements

* Run ```pip install -r requirements.txt```  to install all dependencies

### Data

* The training and testing data can be found [here](https://kelvins.esa.int/collision-avoidance-challenge/data/).
* Download the training data and add it to the folder ```/data/training``` as ```train_data.csv```
* Download the testing data and add it to the folder ```/data/testing``` as ```test_data.csv```

## Usage

After installation and data setup, you can use the provided modules and scripts for various tasks such as data
preprocessing, model training, and evaluation. Here's a brief overview of how to utilize the different components:

* **Data Preprocessing**: Utilize the functions in `src/utils`
  for [loading](src/utils/data_loading.py), [filtering](src/utils/filtering.py), [encoding labels](src/utils/label_encoding.py),
  and [calculating statistics](src/utils/calculate_statistics.py) on your dataset.
* **Exploratory Data Analysis (EDA)**: Refer to the Jupyter notebook `notebooks/exploratory/visualization.ipynb` for
  [visualizing and understanding](notebooks/exploratory/visualization.ipynb) the dataset.
* **Feature Engineering**: Explore and [compare](notebooks/engineering/compare_selected_features_nan.ipynb) feature
  selection techniques provided in the `notebooks/engineering` directory,
  including [dropping](notebooks/engineering/feature_selection_without_nan.ipynb)
  or [imputing](notebooks/engineering/feature_selection_with_nan.ipynb) NaN values,
  and [Principal Component Analysis (PCA)](notebooks/engineering/pca.ipynb).
* **Modeling and Training**: Use
  the [Neural Network Notebook](notebooks/modeling/neural_network_model.ipynb) `notebooks/modeling/neural_network_model.ipynb`
  for building,
  training, and evaluating your machine learning models.
