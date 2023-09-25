import numpy as np
import time
import pickle
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import pdb

resultDir = "/hdd/opportunity/64/results/"


#-------------------------------------------------------------------------------------------------------
# Hyper-parameters : TODO : modify if needed
#-------------------------------------------------------------------------------------------------------

# Path to the files containing training and testing features and labels
best = True

if best:
	trainingDataPath = resultDir+"norm_training_features_matrix.npy"
	trainingLabelsPath = resultDir+"training_labels.npy"
	testingDataPath = resultDir+"norm_testing_features_matrix.npy"
	testingLabelsPath = resultDir+"testing_labels.npy"
else:
	testingDataPath = '/home/prg/programming/python/OPPORTUNITY/dnn_features/dnnFeatures_testing.npy'
	testingLabelsPath = '/home/prg/programming/python/OPPORTUNITY/dnn_features/dnnLabels_testing.npy'
	trainingDataPath = '/home/prg/programming/python/OPPORTUNITY/dnn_features/dnnFeatures_training.npy'
	trainingLabelsPath = '/home/prg/programming/python/OPPORTUNITY/dnn_features/dnnLabels_training.npy'


# Soft-margin parameter
C = [2**e for e in range(-6,6)]
#C = [2**(-4)]
#C = [2]


#-------------------------------------------------------------------------------------------------------
# Function svmClassify: returns classification results using linear SVM
#-------------------------------------------------------------------------------------------------------
def svmClassify(
	trainingDataPath=trainingDataPath,
	trainingLabelsPath=trainingLabelsPath,
	testingDataPath=testingDataPath,
	testingLabelsPath=testingLabelsPath,
	C=C
	):

    start = time.time()

    # Load both training and testing sets
    print('-------------------------------------------------------')
    print('Loading the handcrafted features ...')
    print('-------------------------------------------------------')
    trainingData = np.load(trainingDataPath)
    trainingLabels = np.load(trainingLabelsPath)
    testingData = np.load(testingDataPath)
    testingLabels = np.load(testingLabelsPath)


    # Train the linear SVM model
    for idx in range(len(C)):
	    print('Training the model with C = %.4f' % (C[idx]))
	    classfier = LinearSVC(C=C[idx])
	    classfier.fit(trainingData,trainingLabels)

	    # Evaluate the model on the testing set
	    print('   Evaluating the model')
	    estimatedLabels = classfier.predict(testingData)

	    # Compute the accuracy, weighted F1-score and average F1-score
	    accuracy = accuracy_score(testingLabels,estimatedLabels)
	    weightedF1 = f1_score(testingLabels,estimatedLabels,average='weighted')
	    averageF1 = f1_score(testingLabels,estimatedLabels,average='macro')
	    #confMat = confusion_matrix(testingLabels,estimatedLabels,labels=labelsTable.values())
	    allF1Scores = f1_score(testingLabels,estimatedLabels,average=None)

	    # Print results
	    print('   Test accuracy = %.2f %%' % (accuracy*100))
	    print('   Weighted F1-score = %.4f' % (weightedF1))
	    print('   Average F1-score = %.4f' % (averageF1))
	    print('   All F1-scores:')
	    print(allF1Scores)
	    print('-------------------------------------------------------')
	    #print('Confusion matrix:')
	    #print(confMat)
	    #print('-------------------------------------------------------')
     
    end = time.time()
    print('Script ran for %.2f seconds' % (end-start))

#-------------------------------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	svmClassify()
