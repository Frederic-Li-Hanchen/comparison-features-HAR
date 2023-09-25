import numpy as np

resultDir = "/hdd/opportunity/64/results/"


def normalize_TrainingData(trainingDataP):
    trainingDataP = np.transpose(trainingDataP)
    rows, cols = np.shape(trainingDataP)
    print(rows, cols)
    for i in range(rows):
        trainingDataP[i] = (trainingDataP[i]-trainingDataP[i].min())/(trainingDataP[i].max()-trainingDataP[i].min())

    trainingDataP = np.transpose(trainingDataP)
    return trainingDataP
    
    
    
def normalize_TestingData(trainingData, testingDataP):
    trainingData = np.transpose(trainingData)
    testingDataP = np.transpose(testingDataP)
    rows, cols = np.shape(testingDataP)
    print(rows, cols)
    for i in range(rows):
        testingDataP[i] = (testingDataP[i]-trainingData[i].min())/(trainingData[i].max()-trainingData[i].min()) # maximum and minium will be used from training data

    trainingData = np.transpose(trainingData)
    testingDataP = np.transpose(testingDataP)
    return testingDataP




##Training Data
trainingData = np.load(resultDir+"training_features_matrix.npy")

print("Training data is loaded...")
print("maxTraining:", trainingData.max())
print("min:", trainingData.min())

normalizedTrainingData = normalize_TrainingData(trainingData)
np.save(resultDir+"norm_training_features_matrix.npy", normalizedTrainingData)
print("Training data is normalized...")
print("normalizedMaxTraining:", normalizedTrainingData.max())
print("normalizedMinTraining:", normalizedTrainingData.min())

##Testing Data
trainingData = np.load(resultDir+"training_features_matrix.npy") ## we need to load original data again due to deep/shallow/copy issue
testingData = np.load(resultDir+"testing_features_matrix.npy")

print("Training and Testing data are loaded...")
print("maxTraining:", trainingData.max())
print("min:", trainingData.min())
print("maxTesting:", testingData.max())
print("min:", testingData.min())

normalizedTestingData = normalize_TestingData(trainingData, testingData)
np.save(resultDir+"norm_testing_features_matrix.npy", normalizedTestingData)
print("Testing data is normalized...")
print("normalizedMaxTesting:", normalizedTestingData.max())
print("normalizedMinTesting:", normalizedTestingData.min())
