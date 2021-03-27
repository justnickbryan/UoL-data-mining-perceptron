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
        _errors (:obj: 'list' of :obj: 'int'): a list of integers corresponding to the number of misclassifcations
            (errors) during each epoch of the training.
        _accuracy (float):
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

        return "Perceptron (Training): Epochs = {self.epochs}, Random Seed = {self.seed}, Weights = {self._weights}, Bias = {self._bias},\
            \nErrors = {self._errors}, Accuracy = {self._accuracy:.1f}%\n".format(self=self)


    def train(self, records, labels):
        """Trains the Perceptron for binary classification using a labelled dataset.

        Args:
            records (:obj: 'array' of :obj: 'float'): n x m dimensional array of n records (rows) and m features
                (columns) from the training data.
            labels (:obj: 'array' of :obj: 'int'): n dimensional vector of true class labels (as binary values +1 or -1)
                corresponding to each record in the training data.
        """

        print("\nTRAINING MODE\nInitial")

        # Initialise weights as vector of zeros.
        self._weights = np.zeros((records.shape[1]))

        # Initialise bias as zero.
        self._bias = 0.0

        # Initialise random number generator.
        rng = np.random.default_rng(seed = self.seed)

        # Initialise the errors as an empty list.
        self._errors = []

        # Initialise accuracy as zero.
        self._accuracy = 0.0

        print(self)
        # predictedLabelsList = []

        # Iterate over entire training dataset the number of times given by the epochs value.
        for epoch in range(1, self.epochs + 1):
            # print("EPOCH", epoch)

            # epochPredictedLabels = []

            # Initialise errors for current epoch as zero.
            epochErrors = 0

            # Randomise order of dataset in each epoch.
            records, labels = self.combinedShuffle(rng, records, labels)
            # trueLabelsList = labels.tolist()

            # Iterate over each record and corresponding label in the training dataset.
            for record, label in zip(records, labels):

                activationScore = self.adder(record)
                predictedLabel = self.output(activationScore)
                
                # epochPredictedLabels.append(predictedLabel)

                # Updates weights and bias only if misclassification occurs, i.e. when the product of predicted label
                #   and true label = -1 (e.g. predicted label = -1, true label = +1).
                if (predictedLabel * label) <= 0:
                    epochErrors += 1
                    self._weights = self._weights + (label * record)
                    self._bias = self._bias + label

            # predictedLabelsList.append(epochPredictedLabels)
            self._errors.append(epochErrors)
            self._accuracy = self.evaluation(epochErrors, labels.shape[0])
            # print("Weights =", self._weights, ", Bias =", self._bias)
            # print("Errors =", epochErrors, ", Accuracy =", self._accuracy)

        print("Final")
        print(self)
        return self


    def combinedShuffle(self, rng, records, labels):
        """Shuffles records and labels using the same randomised permutation.
        
        Args:
            rng (:class: 'numpy.random._generator.Generator'): random number generator.
            records (:obj: 'array' of :obj: 'float'): n x m dimensional array of n records (rows) and m features
                (columns) from the training data.
            labels (:obj: 'array' of :obj: 'int'): n dimensional vector of true class labels (as binary values +1 or -1)
                corresponding to each record in the training data.

        Returns:
            shuffledRecords
            shuffledLabels
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


    def evaluation(self, errors, n):
        """Evaluate performance of classification.
        """

        # Accuracy = TP + TN / n, where n is the total number of classifications (TP + TN + FP + FN)
        # FP + FN = errors
        tpAndTn = n - errors
        accuracy = (tpAndTn / n) * 100
        return accuracy

    def test(self, records, labels):
        """Use the trained Perceptron for binary classification of test data.

        Args:
            records (:obj: 'array' of :obj: 'float'): n x m dimensional array of n records (rows) and m features
                (columns) from the test data.
            labels (:obj: 'array' of :obj: 'int'): n dimensional vector of true class labels (as binary values +1 or -1)
                corresponding to each record in the training data.
        """
        
        print("TEST MODE")

        errors = 0

        for record, label in zip(records, labels):

            activationScore = self.adder(record)
            predictedLabel = self.output(activationScore)

            if (predictedLabel * label) <= 0:
                errors += 1
            
            # print("Record = {}, Predicted Label = {}, True Label = {}".format(record, predictedLabel, label))
            
        accuracy = self.evaluation(errors, labels.shape[0])
        print("Perceptron (Testing): Errors = {}, Accuracy = {:.1f}%".format(errors, accuracy))


# Use main() method for implementation of Perceptron for classification tasks.
def main():
    print("Script running in source file")

    ## Set-up ##

    # Load training and test data from .data files and format in numpy arrays.
    trainingArray = np.loadtxt("Data/train.data", dtype = str, delimiter = ",")
    testArray = np.loadtxt("Data/test.data", dtype = str, delimiter = ",")

    # Assign arrays for the records and labels of the training data.
    trainingRecords = trainingArray[:,:4].astype("float64")
    trainingLabels = trainingArray[:,4]

    # Assign arrays for the records and labels of the test data.
    testRecords = testArray[:,:4].astype("float64")
    testLabels = testArray[:,4]

    
    ## Use the binary perceptron to train classifiers to discriminate between two classes ##

    ### Class 1 vs Class 2 ###
    print("\n\n# Class 1 vs Class 2 #")

    # Select records from classes 1 and 2.
    class1And2Records = trainingRecords[:80]
    class1And2TestRecords = testRecords[:20]

    # Assign binary labels for classes 1 and 2, where class 1 records are positive and class 2 are negative.
    class1And2Labels = np.where(trainingLabels[:80] == "class-1", 1, -1)
    class1And2TestLabels = np.where(testLabels[:20] == "class-1", 1, -1)

    # Initialise an instance of the Perceptron for discriminating between class 1 and 2.
    classifier1Vs2 = Perceptron()

    # Train the Perceptron.
    classifier1Vs2.train(class1And2Records, class1And2Labels)

    # Test the Perceptron.
    classifier1Vs2.test(class1And2TestRecords, class1And2TestLabels)



    ### Class 2 vs Class 3 ###
    print("\n\n# Class 2 vs Class 3 #")

    # Select records from classes 2 and 3.
    class2And3Records = trainingRecords[40:]
    class2And3TestRecords = testRecords[10:]

    # Assign binary labels for classes 2 and 3, where class 2 records are positive and class 3 are negative.
    class2And3Labels = np.where(trainingLabels[40:] == "class-2", 1, -1)
    class2And3TestLabels = np.where(testLabels[10:] == "class-2", 1, -1)

    # Initialise an instance of the Perceptron for discriminating between class 2 and 3.
    classifier2Vs3 = Perceptron()

    # Train the Perceptron.
    classifier2Vs3.train(class2And3Records, class2And3Labels)

    # Test the Perceptron.
    classifier2Vs3.test(class2And3TestRecords, class2And3TestLabels)



    ### Class 1 vs Class 3 ###
    print("\n\n# Class 1 vs Class 3 #")

    # Select records from classes 1 and 3.
    class1Records = trainingRecords[:40]
    class3Records = trainingRecords[80:]
    class1And3Records = np.concatenate((class1Records, class3Records))

    class1TestRecords = testRecords[:10]
    class3TestRecords = testRecords[20:]
    class1And3TestRecords = np.concatenate((class1TestRecords, class3TestRecords))

    class1Labels = np.ndarray.flatten(trainingLabels[:40])
    class3Labels = np.ndarray.flatten(trainingLabels[80:])
    class1And3Labels = np.concatenate((class1Labels, class3Labels))

    class1TestLabels = np.ndarray.flatten(testLabels[:10])
    class3TestLabels = np.ndarray.flatten(testLabels[20:])
    class1And3TestLabels = np.concatenate((class1TestLabels, class3TestLabels))

    # Assign binary labels for classes 1 and 3, where class 1 records are positive and class 3 are negative.
    class1And3Labels = np.where(class1And3Labels == "class-1", 1, -1)
    class1And3TestLabels = np.where(class1And3TestLabels == "class-1", 1, -1)

    # Initialise an instance of the Perceptron for discriminating between class 1 and 3.
    classifier1Vs3 = Perceptron()
    
    # Train the Perceptron.
    classifier1Vs3.train(class1And3Records, class1And3Labels)

    # Test the Perceptron.
    classifier1Vs3.test(class1And3TestRecords, class1And3TestLabels)


# Only performs classification tasks above if script has not been imported.
if __name__ == "__main__":
    main()