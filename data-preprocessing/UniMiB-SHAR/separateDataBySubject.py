import numpy as np
import scipy.io as scio
import sys


#-------------------------------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------------------------------
### Hyper-parameters
pathToData = '/hdd/datasets/UniMiB-SHAR/raw/data/acc_data.mat'
pathToLabels = '/hdd/datasets/UniMiB-SHAR/raw/data/acc_labels.mat'

savePath = '/hdd/datasets/UniMiB-SHAR/windows_labels/30_CV_folds/'

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


### Loop on the different subjects of the dataset
nbSubjects = len(set(labels[:,1]))
print('Computing the Leave-One-Out Cross Validation folds for %d subjects' % (nbSubjects))

for subjectId in range(1,nbSubjects+1):
	print('Computing fold for subject %d/%d' % (subjectId,nbSubjects))

	# Determination of the total number of readings for the current subject (for a static allocation of result arrays)
	nbExInFold = 0

	for idx in range(nbReadings):
		if labels[idx,1] == subjectId:
			nbExInFold += 1

	print('   %d examples in fold' % (nbExInFold))

	# Allocation of the data and labels arrays
	foldData = np.zeros((nbExInFold,timeWindow,nbSensors), dtype=np.float32)
	foldLabels = np.zeros((nbExInFold),dtype=int)

	foldIdx = 0

	for idx in range(nbReadings):
		dataFrame = data[idx]
		if labels[idx,1] == subjectId:
			foldData[foldIdx] = np.reshape(dataFrame,(timeWindow,nbSensors))
			foldLabels[foldIdx] = labels[idx,0]-1 # Note: labels cast in the range [0,N-1] with N number of classes
			foldIdx += 1
    
	# Saving results for the current fold
	np.save(savePath+str(subjectId)+'_data.npy',foldData)
	np.save(savePath+str(subjectId)+'_labels.npy',foldLabels)	
