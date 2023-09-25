import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# ---------------------------------------------------------------------------------------------
# Paths to SVM scores and labels -> TODO: modify accordingly
# ---------------------------------------------------------------------------------------------

# Ground truth labels
trueLabels = np.load('trueLabels.npy')

# DNN scores
# NOTE: can be obtained by uncommenting the relevant lines in svmClassify.py
dnnScores = np.load('svmScores_dnn.npy')

# Codebook scores
# NOTE: can be obtained by uncommenting the relevant lines in svmClassify.py
codebookScores = np.load('svmScores_codebook_soft_w16_l1_wn2048_s4_C32.npy')

# Number of classes
nbClasses = 18 # Change to 18 for OPPORTUNITY, 15 for UniMiB-SHAR


# ---------------------------------------------------------------------------------------------
# Computation of estimated labels
# ---------------------------------------------------------------------------------------------
fusedScores = (dnnScores+codebookScores)/2

# Compute the estimated labels for each method
nbExamples = fusedScores.shape[0]

fusionEstLabels = np.zeros((nbExamples),dtype=int)
dnnEstLabels = np.zeros((nbExamples),dtype=int)
codebookEstLabels =  np.zeros((nbExamples),dtype=int)

for idx in range(nbExamples):
	fusionEstLabels[idx] = np.argmax(fusedScores[idx])
	codebookEstLabels[idx] = np.argmax(codebookScores[idx])
	dnnEstLabels[idx] = np.argmax(dnnScores[idx])


# ---------------------------------------------------------------------------------------------
# Displaying results
# ---------------------------------------------------------------------------------------------

# Performance metrics computations for DNN
dnnAccuracy = accuracy_score(trueLabels,dnnEstLabels)
dnnWeightedF1 = f1_score(trueLabels,dnnEstLabels,average='weighted')
dnnAverageF1 = f1_score(trueLabels,dnnEstLabels,average='macro')
dnnConfMat = confusion_matrix(trueLabels,dnnEstLabels,labels=list(range(nbClasses)))
dnnAllF1Scores = f1_score(trueLabels,dnnEstLabels,average=None)
# Print results
print('****************************************************************************')
print('                               DNN results')
print('****************************************************************************')
print('   Test accuracy = %.2f %%' % (dnnAccuracy*100))
print('   Weighted F1-score = %.4f' % (dnnWeightedF1))
print('   Average F1-score = %.4f' % (dnnAverageF1))
print('   All F1-scores:')
print(dnnAllF1Scores)
print('-------------------------------------------------------')
print('Confusion matrix:')
print(dnnConfMat)
print('-------------------------------------------------------')


# Performance metrics computations for codebooks
codebookAccuracy = accuracy_score(trueLabels,codebookEstLabels)
codebookWeightedF1 = f1_score(trueLabels,codebookEstLabels,average='weighted')
codebookAverageF1 = f1_score(trueLabels,codebookEstLabels,average='macro')
codebookConfMat = confusion_matrix(trueLabels,codebookEstLabels,labels=list(range(nbClasses)))
codebookAllF1Scores = f1_score(trueLabels,codebookEstLabels,average=None)
# Print results
print('****************************************************************************')
print('                           Codebook results')
print('****************************************************************************')
print('   Test accuracy = %.2f %%' % (codebookAccuracy*100))
print('   Weighted F1-score = %.4f' % (codebookWeightedF1))
print('   Average F1-score = %.4f' % (codebookAverageF1))
print('   All F1-scores:')
print(codebookAllF1Scores)
print('-------------------------------------------------------')
print('Confusion matrix:')
print(codebookConfMat)
print('-------------------------------------------------------')


# Performance metrics computations for fusion
fusionAccuracy = accuracy_score(trueLabels,fusionEstLabels)
fusionWeightedF1 = f1_score(trueLabels,fusionEstLabels,average='weighted')
fusionAverageF1 = f1_score(trueLabels,fusionEstLabels,average='macro')
fusionConfMat = confusion_matrix(trueLabels,fusionEstLabels,labels=list(range(nbClasses)))
fusionAllF1Scores = f1_score(trueLabels,fusionEstLabels,average=None)
# Print results
print('****************************************************************************')
print('                               Fusion results')
print('****************************************************************************')
print('   Test accuracy = %.2f %%' % (fusionAccuracy*100))
print('   Weighted F1-score = %.4f' % (fusionWeightedF1))
print('   Average F1-score = %.4f' % (fusionAverageF1))
print('   All F1-scores:')
print(fusionAllF1Scores)
print('-------------------------------------------------------')
print('Confusion matrix:')
print(fusionConfMat)
print('-------------------------------------------------------')