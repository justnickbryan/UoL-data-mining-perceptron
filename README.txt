# COMP527 CA1 - PERCEPTRON

Assignment 1 for module COMP527 to implement a Perceptron algorithm in Python.

## To Do List

* Amend printed output of the __str__ method to only show key attributes of the Perceptron: epochs, random seed, weights,
    bias, multi-class classification T/F and L2 regularisation coefficient.

* Move printout of training errors and accuracy to a separate print function under the training method.
   
* Improve evaluation method for Multiclass_Perceptron class

* Update test method for Multiclass_Perceptron class

* Add L2 regularisation Perceptron instances to the main() method.


## Table of Contents
1. General Information
2. Installation
3. Usage
4. Author

### 1. General Information

The Perceptron will be used for binary classification tasks on a dataset.
Multi-class classification will also be demonstrated using a 1-vs-rest approach.
L2 regularisation will also be used.


### 2. Installation

Use the package manager (pip) to install numpy.

```bash
pip install numpy
```


### 3. Usage

In order to successfully run the analysis, ensure that the data files "train.data" and "test.data" are stored in a
subdirectory "Data" of the directory containing the script file CA1_Perceptron.py.
    For example, this directory structure is demonstrated in the following GitHub repository:
    https://github.com/justnickbryan/UoL-data-mining-perceptron


### 4. Author

Nick Bryan,
Student ID:201531951,
MSc Data Science & AI