# COMP527 CA1 - PERCEPTRON

Assignment 1 for module COMP527 to implement a Perceptron algorithm in Python.

## To Do List

* Amend printed output of the __str__ method to only show key attributes of the Perceptron: epochs, random seed, weights,
    bias, multi-class classification T/F and L2 regularisation coefficient.

* Move printout of training errors and accuracy to a separate print function under the training method.

* Add 1 vs Rest input datasets and Perceptron instances to main() method.

* Remove index column from self._trainConfidence array, so that it is ordered values only.

* Convert labels dataset to integer values:
    for label in labels:
        if label == 'class-1':
            label = 1
        elif label == 'class-2':
            label = 2
        elif label == 'class-3':
            label = 3

* Complete train method for Multiclass_Perceptron class by comparing labels of dataset with argmax labels:
    errors = 0
    for predictedLabel, trueLabel in zip(predictedLabels, labels):
        if predictedLabel != trueLabel:
            errors += 1
    
* Add evaluation method for Multiclass_Perceptron class

* Add test method for Multiclass_Perceptron class

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