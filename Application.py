from Data import Data
from Classification import kNN

def main():
    
    # load the datasets    
    # Update the paths of the datasets based on your directory structure
    inputFile1 = 'iris_train.csv'
    inputFile2 = 'iris_test.csv'    
    trainData = Data(inputFile=inputFile1,reference='Species')
    testData = Data(inputFile=inputFile2, reference='Species')    
        
    pl_accuracy, pw_accuracy, sl_accuracy, sw_accuracy = findIndividualAccuracies(trainData,testData)
    min_feature, max_feature = findMinMaxFeatures(pl_accuracy, pw_accuracy, sl_accuracy, sw_accuracy)
    
    knn_predictions,final_accuracy = findOveralAccuracy(trainData,testData)
        
    visualizePredictions(testData,knn_predictions)
    
    print('Prediction accuracy using Petal length = ',round(pl_accuracy,3))
    print('Prediction accuracy using Petal width = ',round(pw_accuracy,3))
    print('Prediction accuracy using Sepal length = ',round(sl_accuracy,3))
    print('Prediction accuracy using Sepal width = ',round(sw_accuracy,3))
    print(min_feature,'has the minimum and',max_feature,'has the maximum accuracy')
    print('Prediction accuracy using all the features = ',round(final_accuracy,3))
        
def findIndividualAccuracies(trainData,testData):
    """
    Find the accuracy of predicting species 
    in iris_test.csv using 
    a. Petal length
    b. Petal width
    c. Sepal length
    d. Sepal width
    
    """
    #creates an object from knn to get one feature 
    Petal_Length = kNN(trainData, 'Petal length')
    
    Petal_Width = kNN(trainData, 'Petal width')
    
    Sepal_Length = kNN(trainData, 'Sepal length')
    
    Sepal_Width = kNN(trainData, 'Sepal width')
    
    
    pl_predict = Petal_Length.classify(testData, k = 5)
    
    pw_predict = Petal_Width.classify(testData, k = 5)
    
    sl_predict = Sepal_Length.classify(testData, k = 5)
    
    sw_predict = Sepal_Width.classify(testData, k = 5)
    
    reference_dictionary = testData.dataDict[testData.reference]
    
    #finds the accuracy for each feature
    pl_accuracy = 100*sum((reference_dictionary== pl_predict)/len(pl_predict))
    
    pw_accuracy = 100*sum((reference_dictionary== pw_predict)/len(pw_predict))
    
    sl_accuracy = 100*sum((reference_dictionary== sl_predict)/len(sl_predict))
    
    sw_accuracy = 100*sum((reference_dictionary== sw_predict)/len(sw_predict))
    
    return pl_accuracy, pw_accuracy, sl_accuracy, sw_accuracy

def findMinMaxFeatures(pl_accuracy, pw_accuracy, sl_accuracy, sw_accuracy):
    """
    Which features have the best and the worst accuracies?
    """
    
    All_Accuracies = {pl_accuracy: 'Petal length', 
                  pw_accuracy: 'Petal width', 
                  sl_accuracy: 'Sepal length', 
                  sw_accuracy: 'Sepal width'}
    
    worst_accuracy = All_Accuracies[min(All_Accuracies.keys())]
    
    best_accuracy = All_Accuracies[max(All_Accuracies.keys())]

    return worst_accuracy, best_accuracy


def findOveralAccuracy(trainData,testData):
    """
    What is the accuracy of predicting species in iris_test.csv 
    using all four features? 
    """
    kNNClassifier = kNN(trainData)
    
    All_Predictions = kNNClassifier.classify(testData,k=5)
    
    reference_dictionary = testData.dataDict['Species']

    Overall_Accuracy = 100*sum(reference_dictionary== All_Predictions)/len(All_Predictions)
    
    return All_Predictions, Overall_Accuracy


def visualizePredictions(testData,knn_predictions):
    """
    Visualize Petal length vs. Petal width
    """
    testData.visualize.scatterPlot('Petal length','Petal width')
    testData.dataDict[testData.reference] = knn_predictions
    testData.visualize.scatterPlot('Petal length','Petal width')

    pass


if __name__ == "__main__": main()
     
        
     