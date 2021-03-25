# COMP527 CA1 - Perceptron
# Nick Bryan (ID: 201531951)


import numpy as np


class Perceptron:
    """Perceptron class for training and testing a binary classifier.
    
    Attributes:
        epochs (int): total number of iterations over the entire set of training data.
        seed (int): seed for reproducible random number generation for data shuffling.
        _weights (:obj: 'array' of :obj: 'float'): m x 1 dimensional array (vector) of m weights (initialised as 0)
            corresponding to the m features of each record.
        _bias (float): input bias.
        _errors (:obj: 'list' of :obj: 'int'): 
    """

    def __init__(self, theEpochs = 20, theSeed = 3):
        """Initialiser creates an instance of the Perceptron.

        Args:
            theEpochs (int, optional): total number of iterations over the entire set of training data. Defaults to 20.
            theSeed (int, optional): seed for reproducible random number generation for data shuffling. Defaults to 20.
        """

        self.epochs = theEpochs
        self.seed = theSeed


    def __str__(self):
        """Returns string representation of Perceptron instance.
                
        Returns:
            str: string representation of Perceptron instance.
        """

        return "Perceptron: Epochs = {self.epochs}, Seed = {self.seed}".format(self=self)


    def train(self, records, labels):
        """Trains the Perceptron for binary classification using a labelled dataset.

        Args:
            records (:obj: 'array' of :obj: 'float'): n x m dimensional array of n records (rows) and m features
                (columns) from the training data.
            labels (:obj: 'array' of :obj: 'int'): n dimensional vector of true class labels (as binary values +1 or -1)
                corresponding to each record in the training data.
        """

        # Initialise weights as vector of zeros.
        self._weights = np.zeros((records.shape[1]))
        print("Initial weights =", self._weights)

        # Initialise bias as zero.
        self._bias = 0.0

        # Initialise random number generator.
        rng = np.random.default_rng(seed = self.seed)
        print("rng = ", rng, ", type = ", type(rng))

        self._errors = []

        # Iterate over entire training dataset the number of times given by the epochs value.
        for epoch in range(self.epochs):
            print("EPOCH:", epoch)

            # Initialise errors as zero for current epoch.
            epochErrors = 0

            # Randomise order of dataset in each epoch.
            records, labels = self.combinedShuffle(rng, records, labels)

            # Iterate over each record and corresponding label in the training dataset.
            for record, label in zip(records, labels):

                activationScore = self.adder(record)
                predictedLabel = self.output(activationScore)

                # Updates weights and bias only if misclassification occurs, i.e. when the product of predicted label
                #   and true label = -1 (e.g. predicted label = -1, true label = +1).
                if (predictedLabel * label) <= 0:
                    epochErrors += 1
                    self._weights = self._weights + (label * record)
                    self._bias = self._bias + label

            print("Weights =", self._weights)
            print("Epoch Errors =", epochErrors)

            self._errors.append(epochErrors)

        return self._weights, self._bias, self._errors


    def combinedShuffle(self, rng, records, labels):
        """Shuffles records and labels in the same manner
        
        Args:
            rng (:class: 'numpy.random._generator.Generator'): random number generator.
            records (:obj: 'array' of :obj: 'float'): n x m dimensional array of n records (rows) and m features
                (columns) from the training data.
            labels (:obj: 'array' of :obj: 'int'): n dimensional vector of true class labels (as binary values +1 or -1)
                corresponding to each record in the training data.
        """
        
        # Generate a permutation index array corresponding to the size of the dataset.
        permutation = rng.permutation(records.shape[0])
        # Shuffle the arrays by giving the permutation in the square brackets (the indices of the array).
        shuffledRecords = records[permutation]
        shuffledLabels = labels[permutation]
        return shuffledRecords, shuffledLabels


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


    def evaluation(self):
        """Evaluate performance of classification.
        """

        # Accuracy = TP + TN / ALL


    def test(self, records):
        """Use the trained Perceptron for binary classification of test data.

        Args:
            records (:obj: 'array' of :obj: 'float'): n x m dimensional array of n records (rows) and m features
                (columns) from the test data.
        """
        
        for record in records:

            activationScore = self.adder(record)

            predictedLabel = self.output(activationScore)

            print(record, predictedLabel)


# Use main() method for implementation of Perceptron for classification tasks.
def main():
    print("Script running in source file")

    # Load training and test data from .data files and format in numpy arrays
    trainingArray = np.loadtxt("Data/train.data", dtype = str, delimiter = ",")
    testArray = np.loadtxt("Data/test.data", dtype = str, delimiter = ",")

    # Assign arrays for the records and labels of the input training data
    trainingRecords = trainingArray[:,:4].astype("float64")
    trainingLabels = trainingArray[:,4]


    ## Use the binary perceptron to train classifiers to discriminate between two classes

    ### Class 1 vs Class 2

    # records for class 1 and 2
    records = trainingRecords[:80]

    labels = np.where(trainingLabels[:80] == "class-1", 1, -1)

    classifier1Vs2 = Perceptron()
    classifier1Vs2.train(records, labels)

    testRecords = testArray[:20,:4].astype("float64")
    classifier1Vs2.test(testRecords)


# Only performs classification tasks above if script has not been imported.
if __name__ == "__main__":
    main()