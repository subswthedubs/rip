import numpy as np
import Data as Data

class kNN:    
    
    def __init__(self,trainData, features=None):
        """
        Initialize the values of trainX, labels and features used to fit
        the kNN classifier
        """
        self.trainX = trainData.getNumpyMatrix(features)
        self.labels = np.array(trainData.dataDict[trainData.reference])
        self.features = features
        
    def computeDistanceMatrix(self,testX):
        """
        [20 points]
        This method computes pairwise distance between
        every row in test data (testX) and every row 
        in the training data.
        Inputs
            •	testX: numpy matrix corresponding to the 
                test data object. 
        Outputs
            •	Distance matrix, D representing pairwise distance
                between every row in testX to every row in trainX.
        
        The element D(i,j) in the resultant distance matrix, 
        D would correspond to the distance between ith row 
        in testX and jth row in trainX.

        """
        Distance_Matrix = np.zeros((np.size(testX,0),np.size(self.trainX,0)))
        
        for i in range(np.size(testX,0)):
            for j in range(np.size(self.trainX,0)):
                Distance_Matrix[i][j] = np.linalg.norm(testX[i,:] - self.trainX[j,:])
        return Distance_Matrix
              
    def classify(self,testData,k):   
        """
        [30 points]
        This method computes prediction for every row in the test data.
        Hints: 
            •	Refer the pseudocode provided
            •	Call the computeDistanceMatrix method from this method
            •	List of useful numpy matrix operations 
                o	numpy.argsort
                o	numpy.unique
                o	numpy.sum
                o	numpy.argmax
        Inputs
            •	testData: data object corresponding to the test data. 
            •	k: Number of nearest neighbors to consider for prediction
        Outputs
            •	kNN based predictions for each row in the test data.

        """
        Distance_Matrix = self.computeDistanceMatrix(testData.getNumpyMatrix())
        Distance_Matrix_Sorted = np.argsort(Distance_Matrix,1)
        D_Sorted_k_Values = Distance_Matrix_Sorted[:,0:k]
        k_Labels = np.zeros((np.size(D_Sorted_k_Values,0),k))
        
        for i in range(np.size(Distance_Matrix,0)):
            for j in range(k):
                k_Labels[i][j] = self.labels[D_Sorted_k_Values[i][j]] 
        Max_Occurance = np.zeros((np.size(k_Labels,0)))
        k_Labels = k_Labels.astype('int64')
        
        for i in range(np.size(k_Labels,0)):
            Max_Occurance[i] = np.bincount(k_Labels[i]).argmax()
        return Max_Occurance

if __name__ == "__main__":
    # load the data sets
    inputFile1 = 'trainData.csv'
    inputFile2 = 'testData.csv'
    trainData = Data.Data(inputFile=inputFile1,reference='diagnosis') 
    testData = Data.Data(inputFile=inputFile2,reference='diagnosis')
    
    #classify the testData
    kNNClassifier = kNN(trainData)
    predictions = kNNClassifier.classify(testData = testData,k=5)
    referenceColumn = testData.dataDict['diagnosis']
    print('Predictions:', predictions)
    print('Test Data Reference column:', referenceColumn)
    print('Accuracy = ', 100*sum( referenceColumn== predictions)/len(predictions))

    
    
        