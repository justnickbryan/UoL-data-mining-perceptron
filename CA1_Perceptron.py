# COMP527 CA1 - Perceptron
# Nick Bryan (ID: 201531951)


import numpy as np


class Perceptron:
    """Perceptron class for training and testing a binary classifier.
    
    Attributes:
        epochs (int): total number of iterations over the entire set of training data.
        seed (int): seed for reproducible random number generation for data shuffling.
        multiclass (boolean): allows multi-class classification if true, else binary classification only.
        l2Coefficient (float): L2 regularisation coefficient.
        _weights (:obj: 'array' of :obj: 'float'): m x 1 dimensional array (vector) of m weights (initialised as 0)
            corresponding to the m features of each record.
        _bias (float): input bias.
        _trainErrors (:obj: 'list' of :obj: 'int'): list of integers corresponding to the number of misclassifications
            (errors) during each epoch of the training.
        _trainAccuracy (float): percentage accuracy of classification measured during the training of the Perceptron.
        _testErrors (int): the number of misclassifications during testing.
        _testAccuracy (float): percentage accuracy of classification measured during the testing of the Perceptron.
        _trainConfidence (:obj: 'list' of :obj: 'float'): list of confidence scores following the training of the model
            for the One-vs-Rest classification of each training record.
        _testConfidence (:obj: 'list' of :obj: 'float'): list of confidence scores for the One-vs-Rest classification
            of each test record.
    """

    def __init__(self, theEpochs = 20, theSeed = 3, theMulticlass = False, theL2Coefficient = 0.0):
        """Initialiser creates an instance of the Perceptron.

        Args:
            theEpochs (int, optional): total number of iterations over the entire set of training data. Defaults to 20.
            theSeed (int, optional): seed for reproducible random number generation for data shuffling. Defaults to 20.
            multiclass (boolean, optional): allows multi-class classification if true, else binary classification only.
                Defaults to False.
            l2Coefficient (float, optional): L2 regularisation coefficient. Defaults to 0.0.
        """

        self.epochs = theEpochs
        self.seed = theSeed
        self.multiclass = theMulticlass
        self.l2Coefficient = theL2Coefficient


    def __str__(self):
        """Prints to screen a string representation of the Perceptron instance.
                
        Returns:
            str: string representation of Perceptron instance, displaying the attributes of the Perceptron's training.
        """

        return "Perceptron (Training): Epochs = {self.epochs}, Random Seed = {self.seed}, Weights = {self._weights}, Bias = {self._bias},\
            \nTraining Errors = {self._trainErrors}, Training Accuracy = {self._trainAccuracy:.1f}%\n".format(self=self)


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
        self._trainErrors = []

        # Initialise accuracy as zero.
        self._trainAccuracy = 0.0

        # Print initialised state of Perceptron before training.
        print(self)

        # Iterate over entire training dataset the number of times given by the epochs value.
        for epoch in range(1, self.epochs + 1):
            # Initialise errors for current epoch as zero.
            epochErrors = 0

            # Randomise order of dataset in each epoch.
            records, labels, permutation = self.combinedShuffle(rng, records, labels)

            # Initialise empty list for the compilation of confidence scores for each training record.
            confidenceScores = []

            # Iterate over each record and corresponding label in the training dataset.
            for record, label in zip(records, labels):

                # Calculate the activation score and predicted class label.
                activationScore = self.adder(record)
                predictedLabel = self.output(activationScore)

                if epoch == self.epochs: # Only true if current epoch equals the final epoch.
                    # For a binary Perceptron, the confidence score is equal to the activation score.
                    # Append the activation score to the list of confidence scores for the current epoch.
                    confidenceScores.append(activationScore)
                    
                # Updates weights and bias only if misclassification occurs, i.e. when the product of predicted label
                #   and true label equals negative one (e.g. predicted label = -1, true label = +1, product = -1).
                if (predictedLabel * label) <= 0:
                    # Add one to number of errors in current epoch for each misclassifcation.
                    epochErrors += 1

                    # Perform update rule for weights and bias.
                    self._weights = self._weights + (label * record)
                    self._bias = self._bias + label


            if epoch == self.epochs: # Only true if current epoch equals the final epoch.    
                
                # DELETE if surplus
                # indexedLabels = list(zip(permutation, labels))
                # labelArray = np.asarray(indexedLabels)
                # sortedLabels = np.sort(labelArray, axis=0)

                # Iterate over both the permutation array from the final training epoch and the list of confidence
                #   scores, to create a new list of indexed scores in tuples as (dataset index, confidence score).
                indexedConfidence = list(zip(permutation, confidenceScores))

                # Convert the list of tuples into an array.
                confidenceArray = np.asarray(indexedConfidence)

                #   Then sort the scores by ascending index value such that the confidence scores are in the same order
                #   as the original dataset.
                self._trainConfidence = np.sort(confidenceArray, axis=0)


            # Update the list of errors by appending the integer value of the total number of errors in current epoch.
            self._trainErrors.append(epochErrors)
            # Calculate and update the accuracy of the classifier for the current epoch.
            self._trainAccuracy = self.evaluation(epochErrors, labels.shape[0])

        # # Assign the list of confidence scores from the final training epoch to an attribute. 
        # self._trainConfidence = confidenceScores

        print("Final")
        # Print final state of Perceptron following training.
        print(self)
        print(self._trainConfidence)


    def combinedShuffle(self, rng, records, labels):
        """Takes the arrays of records and labels and shuffles both arrays using the same randomised permutation.
        
        Args:
            rng (:class: 'numpy.random._generator.Generator'): random number generator.
            records (:obj: 'array' of :obj: 'float'): n x m dimensional array of n records (rows) and m features
                (columns) from the training data.
            labels (:obj: 'array' of :obj: 'int'): n dimensional vector of true class labels (as binary values +1 or -1)
                corresponding to each record in the training data.

        Returns:
            shuffledRecords (:obj: 'array' of :obj: 'float'): randomly permuted n x m dimensional array of n records
                (rows) and m features (columns) from the training data.
            shuffledLabels (:obj: 'array' of :obj: 'int'): randomly permuted n dimensional vector of true class labels
                (as binary values +1 or -1) corresponding to each record in the training data.
            permutation (:obj: 'array' of :obj: 'int'): 
        """
        
        # Generate a permutation index array corresponding to the size of the dataset.
        permutation = rng.permutation(records.shape[0])

        # Shuffle the two arrays by index using the same random permutation (in the square brackets).
        shuffledRecords = records[permutation]
        shuffledLabels = labels[permutation]

        return shuffledRecords, shuffledLabels, permutation


    def adder(self, record):
        """Represents the summing junction of the Perceptron, which returns the activation score for the input record.
        
        Args:
            record (:obj: 'array' of :obj: 'float'): 1 x m dimensional array (vector) of m features of a record in the
                dataset.
        
        Returns:
            activationScore (float): positive or negative value that is the (inner) dot product of inputs and weights
                added to the bias (or threshold).
        """

        # Inner dot product of input and weight vectors, plus the bias.
        activationScore = np.inner(record, self._weights) + self._bias
        return activationScore


    def output(self, activationScore):
        """Output of the Perceptron, which returns the predicted class label given the activation score of an input.

        Args:
            activationScore (float): positive or negative value that is the (inner) dot product of inputs and weights
                added to the bias (or threshold).

        Returns:
            predictedLabel (int): predicted class label (as binary values +1 or -1) for the input record. 
        """

        # Predicts that the input record is from the positive class if greater than zero, or the negative class if less
        #   than or equal to zero.
        if activationScore > 0:
            predictedLabel = 1  
        else:
            predictedLabel = -1
        
        return predictedLabel


    def evaluation(self, errors, n):
        """Evaluates the performance of the Perceptron classifier by calculating the percentage accuracy.

        Args:
            errors (int): the number of misclassifcations in a single training epoch or a pass over the test data.
            n (int): the total of classifcations performed in the training epoch or test.

        Returns:
            accuracy (float): percentage accuracy of training or test classification as an evaluation of the Perceptron.
        """

        # TP = true positive, TN = true negative
        # Therefore, TP + TN = number of correct classifications.
        # Or equivalently, TP + TN = total number of classifications (n) - number of misclassifications (errors).
        tpAndTn = n - errors

        # Percentage Accuracy = (number of correct classifications / total number of classifications) * 100
        accuracy = (tpAndTn / n) * 100
        return accuracy


    def test(self, records, labels):
        """Uses the Perceptron for the binary classification of test data.

        Note: the Perceptron should be trained on a separate training dataset using the Perceptron.train() method prior
            to the classification of test data.

        Args:
            records (:obj: 'array' of :obj: 'float'): n x m dimensional array of n records (rows) and m features
                (columns) from the test data.
            labels (:obj: 'array' of :obj: 'int'): n dimensional vector of true class labels (as binary values +1 or -1)
                corresponding to each record in the training data.
        """
        
        print("TEST MODE")

        # Initialise the number of errors in test mode as zero.
        self._testErrors = 0

        # Initialise empty list for the compilation of confidence scores for each record of the test dataset.
        self._testConfidence = []

        # Iterate over each record and corresponding true class label in the test dataset.
        for record, label in zip(records, labels):

            # Calculate the activation score and predicted class label.
            activationScore = self.adder(record)
            predictedLabel = self.output(activationScore)

            # For a binary Perceptron, the confidence score is equal to the activation score.
            # Append the activation score to the list of confidence scores for the test dataset.
            self._testConfidence.append(activationScore)

            # Add one to number of errors if misclassification occurs, i.e. when the product of predicted label and true
            #   label equals negative one.
            if (predictedLabel * label) <= 0:
                self._testErrors += 1

        # Calculate and update the accuracy of the classifier.                
        self._testAccuracy = self.evaluation(self._testErrors, labels.shape[0])
        print("Perceptron (Testing): Test Errors = {self._testErrors}, Test Accuracy = {self._testAccuracy:.1f}%".format(self=self))


class Multiclass_Perceptron:
    """Multi-class classification with the binary Perceptron the using the One-vs-Rest approach.

    Attributes:
        binaryPerceptrons (:obj: 'tuple' of :class: 'Perceptron'): instances of the binary Perceptron used as the
            predictive models for multi-class classification with a One-vs-Rest approach.
        _trainErrors (:obj: 'list' of :obj: 'int'): a list of integers corresponding to the number of misclassifications
            (errors) during each epoch of the training.
        _trainAccuracy (float): percentage accuracy of classification measured during the training of the Perceptron.
        _testErrors (int): the number of misclassifications during testing.
        _testAccuracy (float): percentage accuracy of classification measured during the testing of the Perceptron.
    """

    def __init__(self, thePerceptrons):

        self.binaryPerceptrons = thePerceptrons


    def train(self, labels):
        
        predictiveModels = []

        for perceptron in self.binaryPerceptrons:
            model = perceptron._trainConfidence
            predictiveModels.append(model)
        
        modelArray = np.asarray(predictiveModels)
        predictedClass = np.argmax(modelArray, axis = 0)
        print(predictedClass)

        # Add one to number of errors if misclassification occurs, i.e. when the product of predicted label and true
        #   label equals negative one.
        if predictedClass !=  0:
            self._trainErrors += 1


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

    #------------------------------------------------------------------------------------------------------------------#

    ## Task 3 ##
    ## Use the binary perceptron to train classifiers to discriminate between two classes. ##

    ### Class 1 vs Class 2 ###
    print("\n\n# Class 1 vs Class 2 #")

    # Select records from classes 1 and 2.
    class1And2Records = trainingRecords[:80] # Training Input Data 
    class1And2TestRecords = testRecords[:20] # Test Input Data

    # Assign binary labels for classes 1 and 2, where class 1 records are positive and class 2 are negative.
    class1And2Labels = np.where(trainingLabels[:80] == "class-1", 1, -1) # Training Data Labels
    class1And2TestLabels = np.where(testLabels[:20] == "class-1", 1, -1) # Test Data Labels

    # Initialise an instance of the Perceptron for discriminating between class 1 and 2.
    classifier1Vs2 = Perceptron()

    # Train the Perceptron.
    classifier1Vs2.train(class1And2Records, class1And2Labels)

    # # Test the Perceptron.
    # classifier1Vs2.test(class1And2TestRecords, class1And2TestLabels)



    # ### Class 2 vs Class 3 ###
    # print("\n\n# Class 2 vs Class 3 #")

    # # Select records from classes 2 and 3.
    # class2And3Records = trainingRecords[40:] # Training Input Data 
    # class2And3TestRecords = testRecords[10:] # Test Input Data 

    # # Assign binary labels for classes 2 and 3, where class 2 records are positive and class 3 are negative.
    # class2And3Labels = np.where(trainingLabels[40:] == "class-2", 1, -1) # Training Data Labels
    # class2And3TestLabels = np.where(testLabels[10:] == "class-2", 1, -1) # Test Data Labels

    # # Initialise an instance of the Perceptron for discriminating between class 2 and 3.
    # classifier2Vs3 = Perceptron()

    # # Train the Perceptron.
    # classifier2Vs3.train(class2And3Records, class2And3Labels)

    # # Test the Perceptron.
    # classifier2Vs3.test(class2And3TestRecords, class2And3TestLabels)



    # ### Class 1 vs Class 3 ###
    # print("\n\n# Class 1 vs Class 3 #")

    # # Select records from classes 1 and 3. Then concatenate the separate arrays for classes 1 and 3.
    # class1Records = trainingRecords[:40]
    # class3Records = trainingRecords[80:]
    # class1And3Records = np.concatenate((class1Records, class3Records)) # Training Input Data 

    # class1TestRecords = testRecords[:10]
    # class3TestRecords = testRecords[20:]
    # class1And3TestRecords = np.concatenate((class1TestRecords, class3TestRecords)) # Test Input Data 

    # class1Labels = np.ndarray.flatten(trainingLabels[:40])
    # class3Labels = np.ndarray.flatten(trainingLabels[80:])
    # class1And3Labels = np.concatenate((class1Labels, class3Labels)) # Training Data Labels (as strings)

    # class1TestLabels = np.ndarray.flatten(testLabels[:10])
    # class3TestLabels = np.ndarray.flatten(testLabels[20:])
    # class1And3TestLabels = np.concatenate((class1TestLabels, class3TestLabels)) # Test Data Labels (as strings)

    # # Assign binary labels for classes 1 and 3, where class 1 records are positive and class 3 are negative.
    # class1And3Labels = np.where(class1And3Labels == "class-1", 1, -1) # Training Data Labels
    # class1And3TestLabels = np.where(class1And3TestLabels == "class-1", 1, -1) # Test Data Labels

    # # Initialise an instance of the Perceptron for discriminating between class 1 and 3.
    # classifier1Vs3 = Perceptron()
    
    # # Train the Perceptron.
    # classifier1Vs3.train(class1And3Records, class1And3Labels)

    # # Test the Perceptron.
    # classifier1Vs3.test(class1And3TestRecords, class1And3TestLabels)


    #------------------------------------------------------------------------------------------------------------------#


    # ## Task 4 ##
    # ## Extend the binary perceptron to perform multi-class classification using the One-vs-Rest approach. ##

    # print("\n\n# Multi-Class Classification: One vs Rest #")

    # ### Class 1 vs Rest ###

    # # Assign binary labels, where class 1 records are positive and the rest are negative.
    # trainingLabelsClass1 = np.where(trainingLabels == "class-1", 1, -1) # Training Data Labels
    # testLabelsClass1 = np.where(testLabels == "class-1", 1, -1) # Test Data Labels

    # # Initialise an instance of the binary Perceptron for the positive class 1.
    # classifier1VsRest = Perceptron(theMulticlass=True)
    # # Train the Perceptron binary model on the entire training dataset.
    # classifier1VsRest.train(trainingRecords, trainingLabelsClass1)


    # ### Class 2 vs Rest ###

    # # Assign binary labels, where class 2 records are positive and the rest are negative.
    # trainingLabelsClass2 = np.where(trainingLabels == "class-2", 1, -1) # Training Data Labels
    # testLabelsClass2 = np.where(testLabels == "class-2", 1, -1) # Test Data Labels

    # # Initialise an instance of the binary Perceptron for the positive class 2.
    # classifier2VsRest = Perceptron(theMulticlass=True)
    # # Train the Perceptron binary model on the entire training dataset.
    # classifier2VsRest.train(trainingRecords, trainingLabelsClass2)


    # ### Class 3 vs Rest ###

    # # Assign binary labels, where class 3 records are positive and the rest are negative.
    # trainingLabelsClass3 = np.where(trainingLabels == "class-3", 1, -1) # Training Data Labels
    # testLabelsClass3 = np.where(testLabels == "class-3", 1, -1) # Test Data Labels

    # # Initialise an instance of the binary Perceptron for the positive class 3.
    # classifier3VsRest = Perceptron(theMulticlass=True)
    # # Train the Perceptron binary model on the entire training dataset.
    # classifier3VsRest.train(trainingRecords, trainingLabelsClass3)


    # ### Multi-Class Classification ###
    
    # # Initialise an instance of the multi-class Perceptron for evaluation of the One-vs-Rest approach.
    # multiClassifier = Multiclass_Perceptron((classifier1VsRest, classifier2VsRest, classifier3VsRest))
    # # Predict the 




# Only performs classification tasks above if script has not been imported.
if __name__ == "__main__":
    main()