import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from utilitary import labelsTable
from trainDnn import C, dnnFeaturesPath


#-------------------------------------------------------------------------------------------------------
# Function svmClassify: returns classification results using a linear SVM
#-------------------------------------------------------------------------------------------------------
def svmClassify(
	trainingDataPath,
	trainingLabelsPath,
	testingDataPath,
	testingLabelsPath,
	C
	):

    start = time.time()

    # Load both training and testing sets
    print('Loading the data ...')
    trainingData = np.load(trainingDataPath)
    trainingLabels = np.load(trainingLabelsPath)
    testingData = np.load(testingDataPath)
    testingLabels = np.load(testingLabelsPath)

    bestAF1 = 0

    # Train the linear SVM model
    for idx in range(len(C)):
	    print('Training the model with C = %.4f' % (C[idx]))
	    classifier = LinearSVC(C=C[idx])
	    classifier.fit(trainingData,trainingLabels)

	    # Evaluate the model on the testing set
	    print('   Evaluating the model')
	    estimatedLabels = classifier.predict(testingData)

	    # Compute the accuracy, weighted F1-score and average F1-score
	    accuracy = accuracy_score(testingLabels,estimatedLabels)
	    weightedF1 = f1_score(testingLabels,estimatedLabels,average='weighted')
	    averageF1 = f1_score(testingLabels,estimatedLabels,average='macro')
	    confMat = confusion_matrix(testingLabels,estimatedLabels,labels=labelsTable.values())
	    allF1Scores = f1_score(testingLabels,estimatedLabels,average=None)

	    # Print results
	    print('   Test accuracy = %.2f %%' % (accuracy*100))
	    print('   Weighted F1-score = %.4f' % (weightedF1))
	    print('   Average F1-score = %.4f' % (averageF1))
	    print('   All F1-scores:')
	    print(allF1Scores)
	    print('-------------------------------------------------------')
	    print('Confusion matrix:')
	    print(confMat)
	    print('-------------------------------------------------------')

	    if averageF1 > bestAF1: # Optimization in terms of average F1-score
	    	bestAF1 = averageF1
	    	bestAcc = accuracy*100
	    	bestWF1 = weightedF1
	        decisionScores = classifier.decision_function(testingData)
	        debugLabels = estimatedLabels

            #np.save('../fusion/DNN/dnnSvmScores.npy',decisionScores)  # Uncomment to save the SVM scores
            #np.save('../fusion/trueLabels.npy',testingLabels) # Uncomment to save the ground truth labels
     
    end = time.time()
    print('Script ran for %.2f seconds' % (end-start))


#-------------------------------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    print('#############################################################')
    print('SVM training and evaluation ...')
    print('#############################################################')
    svmClassify(trainingDataPath=dnnFeaturesPath+'/dnnFeatures_training.npy',
              trainingLabelsPath=dnnFeaturesPath+'/dnnLabels_training.npy',
              testingDataPath=dnnFeaturesPath+'/dnnFeatures_testing.npy',
              testingLabelsPath=dnnFeaturesPath+'/dnnLabels_testing.npy',
              C=C)