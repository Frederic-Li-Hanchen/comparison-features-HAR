from keras import backend as K

import numpy as np
import os
import sys


#-------------------------------------------------------------------------------------------------------
# UniMiB-SHAR meta-variables
#-------------------------------------------------------------------------------------------------------
timeWindow = 151 # Size of the time window | Note: fixed value
nbSensors = 3 # Number of sensor channels | Note: fixed value
nbClasses = 17 # Number of classes: 9 ADL + 7 fall


#-------------------------------------------------------------------------------------------------------
# Input structure for model params
#-------------------------------------------------------------------------------------------------------
class modelParam():

    def __init__(self,name,params):
        self.name = name # str
        self.params = params # dict

    def getModelName(self):
        print('Model selected: %s' % (self.name))
        

#-------------------------------------------------------------------------------------------------------
# Function shuffleInUnisson : apply the same random permutation to 2 different arrays/vectors
#-------------------------------------------------------------------------------------------------------
def shuffleInUnisson(a,b): # a and b must be vectors or arrays with the same number of lines
    assert len(a) == len(b)
    randomPermutation = np.random.permutation(len(a))
    return a[randomPermutation], b[randomPermutation]


#-------------------------------------------------------------------------------------------------------
# Function precision, recall, fbeta_score and fmeasure
#-------------------------------------------------------------------------------------------------------
def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.

    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=2)


#-------------------------------------------------------------------------------------------------------
# buildCVDataset : build a dataset for a Leave-One-Subject-Out cross validation
# dataPath designates a folder containing the data and labels as .npy files, separated by subjects
# foldSubjectId designates the ID of the subject of the dataset for whom to build the dataset
#-------------------------------------------------------------------------------------------------------
def buildCVDataset(dataPath,foldSubjectId,nbSensors=3,timeWindow=151,nbFolds=30):

    # List the contents of the data folder
    listFolder = os.listdir(dataPath)

    # Separate data and labels
    listData = [name for name in listFolder if '_data.npy' in name]
    listData = sorted(listData)
    listLabels = [name for name in listFolder if '_labels.npy' in name]
    listLabels = sorted(listLabels)

    # Find the number of examples for the training set
    nbExamplesPerFold = np.zeros((nbFolds),dtype=int)

    for idx in range(len(listLabels)):
        if listLabels[idx] != str(foldSubjectId)+'_labels.npy':
            labels = np.load(dataPath+listLabels[idx])
            nbExamplesPerFold[idx] = len(labels)

    nbTrainingEx = np.sum(nbExamplesPerFold)

    # Build the training and testing datasets
    trainingData = np.zeros((nbTrainingEx,timeWindow,nbSensors),dtype=np.float32)
    trainingLabels = -1*np.ones((nbTrainingEx),dtype=int)

    trainingIdx = 0

    for idx in range(len(listData)):
        if listData[idx] != str(foldSubjectId)+'_data.npy':
            trainingData[trainingIdx:trainingIdx+nbExamplesPerFold[idx]] = np.load(dataPath+listData[idx])
            trainingLabels[trainingIdx:trainingIdx+nbExamplesPerFold[idx]] = np.load(dataPath+listLabels[idx])
            trainingIdx += nbExamplesPerFold[idx]
            unique = set(np.load(dataPath+listLabels[idx]))
            #if len(unique) != 17:
            #    print('---------------------------------------------------------------')
            #    print('Not all classes represented for data file ' + listData[idx])
             #   print(unique)
        else:
            testingData = np.load(dataPath+listData[idx])
            testingLabels = np.load(dataPath+listLabels[idx])

    # Shuffle data and labels
    trainingData, trainingLabels = shuffleInUnisson(trainingData,trainingLabels)
    testingData, testingLabels = shuffleInUnisson(testingData,testingLabels)

    return trainingData, trainingLabels, testingData, testingLabels
