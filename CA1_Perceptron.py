# COMP527 CA1 - Perceptron
# Nick Bryan (ID: 201531951)

import numpy as np

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

        return "Perceptron: Epochs = {self.epochs}".format(self=self))


    def trainPerceptron(self):
        """Trains the Perceptron for binary classification using a labelled dataset.

        Args:
            trainingData (array): n x m dimensional array of n records (rows) and m features (columns).
            trainingTrueClass (array): n x 1 dimensional array (vector) of n labels corresponding to each record in the training data.
        """