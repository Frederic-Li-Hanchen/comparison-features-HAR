import numpy as np
import os


resultPath = '/home/prg/programming/python/UniMiB-SHAR/results/' # TODO: change to the path where evaluation metrics for each subject are saved

csvFile = resultPath+'/cv_results.csv'


fullResults = np.zeros((30,3),dtype=np.float32)

for subjectId in range(1,31):
	pathToResults = resultPath+'/eval_metrics_s'+str(subjectId)+'.npy'
	results = np.load(pathToResults)
	fullResults[subjectId-1,0] = results[0]
	fullResults[subjectId-1,1] = results[1]*100
	fullResults[subjectId-1,2] = results[2]*100

print(fullResults)
print('***********************************')
print('Average accuracy = %.2f %%' % (np.mean(fullResults[:,0])))
print('Average AF1 = %.2f %%' % (np.mean(fullResults[:,1])))
print('Average WF1 = %.2f %%' % (np.mean(fullResults[:,2])))
print('***********************************')

np.savetxt(csvFile,fullResults,delimiter=",")
