# COMP527 CA1 - PERCEPTRON

Assignment 1 for module COMP527 to implement a Perceptron algorithm in Python.


## Table of Contents
1. Description
2. Installation
3. Usage
4. Author


### 1. Description

For this assignment, a Perceptron algorithm has been implemented in a class-based structure.
The Perceptron class is used for training and testing binary classification.
The Multiclass_Perceptron class takes trained instances of the Perceptron class as models for multi-class classification
with a one-vs-rest approach.

A training and a test dataset were provided for the tasks of:
    * binary classification,
    * multi-class classification using a one-vs-rest approach,
    * multi-class classification using a one-vs-rest approach with L2-regularisation.

A main() method provides the data collection and preparation for the implementation of the algorithm for these tasks.

Training and test accuracies are reported as a measure of classification performance.


### 2. Installation

Use the package manager (pip) to install numpy.

```bash
pip install numpy
```


### 3. Usage

Python file: "CA1_Perceptron.py"

In order to successfully run the analysis, ensure that the two data files "train.data" and "test.data" are stored in a
subdirectory "Data" of the directory containing the script file "CA1_Perceptron.py".
    For example, this directory structure is demonstrated in the following GitHub repository:
    https://github.com/justnickbryan/UoL-data-mining-perceptron


### 4. Author

Nick Bryan,
Student ID:201531951,
MSc Data Science & AI