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

## Before pushing to   git

* Run ```flake8 .``` for linting
* Run ```black .``` for auto formatting