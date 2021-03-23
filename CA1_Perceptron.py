# COMP527 CA1 - Perceptron
# Nick Bryan (ID: 201531951)

import numpy as np

# load training and test data from .data files and format in numpy arrays
trainingArray = np.loadtxt("Data/train.data", dtype = str, delimiter = ",")

# assign array for the features of the input training data
trainingFeatures = trainingArray[:,:4].astype("float64")

# assign array for the labels of the input training data
trainingLabels = trainingArray[:,4]

# e.g for question 3 part 1
trueLabel = np.where(trainingLabels == 'class-1', -1, 1)

testArray = np.loadtxt("Data/test.data", dtype = str, delimiter = ",")


class Perceptron:
    """Perceptron class for training and testing a binary classifier.
    
    Attributes:
        epochs (int): total number of iterations over the entire set of training data.
        weights (array): m x 1 dimensional array (vector) of m weights (initialised as 0) corresponding to the m features of each record.
        bias (float): input bias.
    """

    def __init__(self, theEpochs = 50):
        """Initialiser creates an instance of the Perceptron.

        Args:
            theEpochs (int, optional): total number of iterations over the entire set of training data. Defaults to 50.
        """

        self.epochs = theEpochs


    def __str__(self):
        """Returns string representation of Perceptron instance.
                
        Returns:
            str: string representation of Perceptron instance.
        """

        return "Perceptron: Epochs = {self.epochs}".format(self=self)


    def train(self):
        """Trains the Perceptron for binary classification using a labelled dataset.

        Args:
            trainingFeatures (array): n x m dimensional array of n records (rows) and m features (columns) from the training data.
            trainingLabel (array): n x 1 dimensional array (vector) of n true class labels corresponding to each record in the training data.
        """

        # assign array for the features of the input training data
        trainingFeatures = trainingArray[:,:4].astype("float64")
        
        # assign array for the labels of the input training data
        trainingLabels = trainingArray[:,4]

        # initialise weights as vector of zeros
        self.weights = np.zeros((trainingFeatures.shape[1],1))

        # initialise bias as zero
        self.bias = 0.0

        for epoch in range(self.epochs):

            for record, label in zip(trainingFeatures, trainingLabels):

                activationScore = np.inner(record, self.weights) + self.bias

                print(activationScore)

                if activationScore > 0:
                    self.predictedClass = 1
                
                else:
                    self.predictedClass = -1
                    

                    if (activationScore * label) <= 0:

                        for weight in range(self.weights.shape[0]):
                            weight = update