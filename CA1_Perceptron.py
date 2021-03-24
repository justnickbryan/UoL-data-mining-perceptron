# COMP527 CA1 - Perceptron
# Nick Bryan (ID: 201531951)

import numpy as np

# load training and test data from .data files and format in numpy arrays
trainingArray = np.loadtxt("Data/train.data", dtype = str, delimiter = ",")
testArray = np.loadtxt("Data/test.data", dtype = str, delimiter = ",")


class Perceptron:
    """Perceptron class for training and testing a binary classifier.
    
    Attributes:
        epochs (int): total number of iterations over the entire set of training data.
        _weights (:obj: 'array' of :obj: 'float'): m x 1 dimensional array (vector) of m weights
            (initialised as 0) corresponding to the m features of each record.
        _bias (float): input bias.
    """

    def __init__(self, theEpochs = 20):
        """Initialiser creates an instance of the Perceptron.

        Args:
            theEpochs (int, optional): total number of iterations over the entire set of training data.
                Defaults to 50.
        """

        self.epochs = theEpochs


    def __str__(self):
        """Returns string representation of Perceptron instance.
                
        Returns:
            str: string representation of Perceptron instance.
        """

        return "Perceptron: Epochs = {self.epochs}".format(self=self)


    def train(self, records, labels):
        """Trains the Perceptron for binary classification using a labelled dataset.

        Args:
            records (:obj: 'array' of :obj: 'float'): n x m dimensional array of n records (rows)
                and m features (columns) from the training data.
            labels (:obj: 'array' of :obj: 'int'): n x 1 dimensional array (vector) of n true class
                labels (as binary values +1 or -1) corresponding to each record in the training data.
        """

        # initialise weights as vector of zeros
        self._weights = np.zeros((records.shape[1]))
        print("Initial weights =", self._weights)

        # initialise bias as zero
        self._bias = 0.0

        for epoch in range(self.epochs):
            print("EPOCH:", epoch)

            for record, label in zip(records, labels):

                activationScore = self.adder(record)
                predictedLabel = self.output(activationScore)

                if (predictedLabel * label) <= 0:
                    self._weights = self._weights + (label * record)
                    self._bias = self._bias + label

            print("Weights =", self._weights)

        return self._weights, self._bias


    def adder(self, record):
        """Summing junction of the Perceptron, which returns the activation score for the input record.
        
        Args:
            record (:obj: 'array' of :obj: 'float'): 1 x m dimensional array (vector) of m features of a record in the 
                dataset.
        
        Returns:
            activationScore (float): positive or negative value that is the (inner) dot product of inputs and weights
                added to the bias (or threshold).
        """

        activationScore = np.inner(record, self._weights) + self._bias
        return activationScore


    def output(self, activationScore):
        """Output of the Perceptron, which returns the predicted class label for the input record.

        Args:
            activationScore (float): positive or negative value that is the (inner) dot product of inputs and weights
                added to the bias (or threshold).

        Returns:
            predictedLabel (int): predicted class label (as binary values +1 or -1) for the input record. 
        """
        if activationScore > 0:
            predictedLabel = 1  
        else:
            predictedLabel = -1
        
        return predictedLabel


    def evaluation(self, predictedLabel, label):
        """Evaluate performance of classification.
        """


    def test(self, records):
        """Use the trained Perceptron for binary classification of test data.

        Args:
            records (:obj: 'array' of :obj: 'float'): n x m dimensional array of n records (rows)
                and m features (columns) from the test data.
        """
        
        for record in records:

            activationScore = self.adder(record)

            predictedLabel = self.output(activationScore)

            print(record, predictedLabel)


# Implementation of Perceptron for classification tasks

## Use the binary perceptron to train classifiers to discriminate between two classes

# assign array for the features of the input training data
trainingRecords = trainingArray[:,:4].astype("float64")
# assign array for the labels of the input training data
trainingLabels = trainingArray[:,4]


### Class 1 vs Class 2

# records for class 1 and 2
records = trainingRecords[:80]

labels = np.where(trainingLabels[:80] == "class-1", 1, -1)

classifier1Vs2 = Perceptron()
classifier1Vs2.train(records, labels)

testRecords = testArray[:20,:4].astype("float64")
classifier1Vs2.test(testRecords)