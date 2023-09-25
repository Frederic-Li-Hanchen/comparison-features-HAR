import numpy as np
import scipy.io as scio
import sys
#from utilitary import shuffleInUnisson


#-------------------------------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------------------------------
### Hyper-parameters
trainingSubjectsId = list(range(1,21))
testingSubjectsId = list(range(21,31))

pathToData = '/hdd/datasets/UniMiB-SHAR/raw/data/acc_data.mat'
pathToLabels = '/hdd/datasets/UniMiB-SHAR/raw/data/acc_labels.mat'

trainingSavePath = '/hdd/datasets/UniMiB-SHAR/windows_labels/training/'
testingSavePath = '/hdd/datasets/UniMiB-SHAR/windows_labels/testing/'

nbSensors = 3
timeWindow = 151

### Loading the matlab files
print('Loading data files ...')
data = scio.loadmat(pathToData)
data = data['acc_data']
labels = scio.loadmat(pathToLabels) # Note: first column = activity label, second column = subject id, third column = number of trials
labels = labels['acc_labels']

nbReadings = data.shape[0]
assert nbReadings == labels.shape[0]
print('   %d total examples loaded' % (nbReadings))


### Determination of the total number of training and testing readings (for a static allocation of result arrays)
print('Computing number of training and testing examples ...')
nbTrainingEx = 0
nbTestingEx = 0

for idx in range(nbReadings):
	if labels[idx,1] in trainingSubjectsId:
		nbTrainingEx += 1
	elif labels[idx,1] in testingSubjectsId:
		nbTestingEx += 1
	else:
		print('Error! Subject ID %d unknown!' % (labels[idx,1]))
		sys.exit()

assert nbTrainingEx + nbTestingEx == nbReadings

print('   %d training examples' % (nbTrainingEx))
print('   %d testing examples' % (nbTestingEx))


### Separation of the training and testing data
print('Building training and testing sets ...')
trainingData = np.zeros((nbTrainingEx,timeWindow,nbSensors), dtype=np.float32)
trainingLabels = np.zeros((nbTrainingEx),dtype=int)
testingData = np.zeros((nbTestingEx,timeWindow,nbSensors), dtype=np.float32)
testingLabels = np.zeros((nbTestingEx),dtype=int)

trainingIdx = 0
testingIdx = 0

for idx in range(nbReadings):
	dataFrame = data[idx]
	if labels[idx,1] in trainingSubjectsId:
		trainingData[trainingIdx] = np.reshape(dataFrame,(timeWindow,nbSensors))
		trainingLabels[trainingIdx] = labels[idx,0]-1 # Note: labels cast in the range [0,N-1] with N number of classes
		trainingIdx += 1
	else:
		testingData[testingIdx] = np.reshape(dataFrame,(timeWindow,nbSensors))
		testingLabels[testingIdx] = labels[idx,0]-1 # Note: labels cast in the range [0,N-1] with N number of classes
		testingIdx += 1

#trainingData, trainingLabels = shuffleInUnisson(trainingData, trainingLabels)
#testingData, testingLabels = shuffleInUnisson(testingData, testingLabels)
print('   Done')


### Saving results
print('Saving results in '+trainingSavePath+' and '+testingSavePath)
np.save(trainingSavePath+'training_data.npy',trainingData)
np.save(trainingSavePath+'training_labels.npy',trainingLabels)
np.save(testingSavePath+'testing_data.npy',testingData)
np.save(testingSavePath+'testing_labels.npy',testingLabels)		
