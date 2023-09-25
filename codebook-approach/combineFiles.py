
import numpy as np
import sys

featDir = "./soft/"
labDir = "../data/windows_labels/64/"

trainingFeatFiles = [
    "S1-ADL1_data.npy", "S1-ADL2_data.npy", "S1-ADL3_data.npy", "S1-Drill_data.npy",
    "S2-ADL1_data.npy", "S2-ADL2_data.npy", "S2-ADL3_data.npy", "S2-Drill_data.npy",
    "S3-ADL1_data.npy", "S3-ADL2_data.npy", "S3-ADL3_data.npy", "S3-Drill_data.npy",
    "S4-ADL1_data.npy", "S4-ADL2_data.npy", "S4-ADL3_data.npy", "S4-Drill_data.npy"
]

testingFeatFiles = [
    "S1-ADL4_data.npy", "S1-ADL5_data.npy",
    "S2-ADL4_data.npy", "S2-ADL5_data.npy",
    "S3-ADL4_data.npy", "S3-ADL5_data.npy",
    "S4-ADL4_data.npy", "S4-ADL5_data.npy",
]

print(">> Loading features for training examples.")
trainingFeats = [np.load(featDir + fn) for fn in trainingFeatFiles]
trainingData = np.vstack(trainingFeats)
trainingData = trainingData.astype(np.float32)
print("---> {0}".format(trainingData.shape))
np.save(featDir + "Features_training.npy", trainingData)

print(">> Loading features for testing examples.")
testingFeats = [np.load(featDir + fn) for fn in testingFeatFiles]
testingData = np.vstack(testingFeats)
testingData = testingData.astype(np.float32)
print("---> {0}".format(testingData.shape))
np.save(featDir + "Features_testing.npy", testingData)
"""
trainingLabFiles = [
    "S1-ADL1_labels.npy", "S1-ADL2_labels.npy", "S1-ADL3_labels.npy", "S1-Drill_labels.npy",
    "S2-ADL1_labels.npy", "S2-ADL2_labels.npy", "S2-ADL3_labels.npy", "S2-Drill_labels.npy",
    "S3-ADL1_labels.npy", "S3-ADL2_labels.npy", "S3-ADL3_labels.npy", "S3-Drill_labels.npy",
    "S4-ADL1_labels.npy", "S4-ADL2_labels.npy", "S4-ADL3_labels.npy", "S4-Drill_labels.npy",
]

testingLabFiles = [
    "S1-ADL4_labels.npy", "S1-ADL5_labels.npy",
    "S2-ADL4_labels.npy", "S2-ADL5_labels.npy",
    "S3-ADL4_labels.npy", "S3-ADL5_labels.npy",
    "S4-ADL4_labels.npy", "S4-ADL5_labels.npy",
]

print(">> Loading labels for training examples.")
trainingLabelsSrc = [np.load(labDir + fn) for fn in trainingLabFiles]
trainingLabels = np.hstack(trainingLabelsSrc)
print("---> {0}".format(trainingLabels.shape))
np.save(labDir + "Labels_training.npy", trainingLabels)

print(">> Loading labels for testing examples.")
testingLabelsSrc = [np.load(labDir + fn) for fn in testingLabFiles]
testingLabels = np.hstack(testingLabelsSrc)
print("---> {0}".format(testingLabels.shape))
np.save(labDir + "Labels_testing.npy", testingLabels)
"""
