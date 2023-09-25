import numpy as np
import math
import os
import scipy.stats 

inputDir = "/hdd/datasets/OPPORTUNITY/windows_labels/64/"
resultDir = "/hdd/opportunity/64/results/"                

def generateAndSaveFeatures():
    outputFileName = ""
    for file in os.listdir(inputDir): 
        if file.endswith('_data.npy'): 
            print('Processing file ' + file + ' ...')
            # File name
            fileName = file.replace('_data.npy','')
            outputFileName = fileName+"_features.npy"

            data = np.load(inputDir+file)
            dims = np.shape(data)
            print(dims)
            features = np.zeros((dims[0], 18, dims[2]))  ## to store 18 handcrafted features
            for i in range(dims[0]): # for each timeWindow (Original)
                for j in range(dims[2]): # for each sensor (Original)
                    d = data[i,:,j]
                    # feature extraction
                    mx = max(d)                                                     #1
                    mn = min(d)                                                     #2
                    avg = np.mean(d)                                                #3
                    std = scipy.stats.tstd(d)                                       #4
                    zc = getZeroCrossings(avg, d)                                   #5
                    pt20 = np.percentile(d,20)                                      #6
                    pt50 = np.percentile(d,50)                                      #7
                    pt80 = np.percentile(d,80)                                      #8
                    interquartile = np.percentile(d,75)-np.percentile(d,25)         #9
                    skewness = scipy.stats.mstats.skew(d)                           #10
                    kurtosis = scipy.stats.mstats.kurtosis(d)                       #11
                    #### auto corelation ####
                    avg = np.mean(d)
                    if avg == mx:   #all data elements are same
                        dm = d
                    else:
                        dm = d-avg
                    nd = np.zeros(len(dm)-1)
                    dms = dm**2
                    dmss = sum(dms)
                    for z in range(len(dm)-1):
                        nd[z] = dm[z]*dm[z+1]
                    if (dmss != 0):
                        acr = sum(nd)/dmss                                          #12
                    else:
                        acr = sum(nd)
                    ###########################
                    meanNormFirstOrder = 0
                    meanNormSecondOrder = 0
                    
                    firstOrDiff = np.diff(d,1)
                    meanFirstOrder = firstOrDiff.mean()                             #13
                    if min(firstOrDiff) != max(firstOrDiff):
                        normalizedMFO = (firstOrDiff-min(firstOrDiff))/(max(firstOrDiff)- min(firstOrDiff))
                        meanNormFirstOrder = normalizedMFO.mean()                   #14
                    elif max(firstOrDiff) != 0:
                        normalizedMFO = firstOrDiff/max(firstOrDiff)
                        meanNormFirstOrder = normalizedMFO.mean()                   #14

                    #meanSecondOrder = np.diff(x,2).mean()                           
                    secondOrDiff = np.diff(d,2)
                    meanSecondOrder = secondOrDiff.mean()                           #15
                    if min(secondOrDiff) != max(secondOrDiff):
                        normalizedMSO = (secondOrDiff-min(secondOrDiff))/(max(secondOrDiff)- min(secondOrDiff))
                        meanNormSecondOrder = normalizedMSO.mean()                  #16
                    elif max(secondOrDiff) != 0:
                        normalizedMSO = secondOrDiff/max(secondOrDiff)
                        meanNormSecondOrder = normalizedMSO.mean()                  #16

                    #### spectral energy and entropy ####
                    f = np.fft.fft(d)  #discrete fourier
                    F = abs(f)
                    sumF = sum(F)
                    if sumF == 0:
                        sumF = 1
                    nF = F/sumF
                    min_nF = 1
                    if (min(nF) != max(nF)) and (min(nF) != 0):  #if nF contains only zeros then min() at the next line returns empty set, So i used this condition to avoid empty set error.
                        min_nF = min(m for m in nF if m > 0)

                    spectralEnergy = sum(np.square(F)) #spectral energy                 #17
                    nSpectralEnergy = sum(np.square(nF)) #normalized spectral energy    
                    #logF = np.log10(nF)
                    #logF = np.log(nF)
                    logF = np.log((nF+min_nF))
                    #spectralEntropy = np.nan_to_num(-1*sum(nF*logF)) #spectral entropy  #18
                    spectralEntropy = -1*sum(nF*logF) #spectral entropy                 #18
                    
                    
                    features[i][0][j] = mx
                    features[i][1][j] = mn
                    features[i][2][j] = avg
                    features[i][3][j] = std
                    features[i][4][j] = zc
                    features[i][5][j] = pt20
                    features[i,6,j] = pt50
                    features[i,7,j] = pt80
                    features[i,8,j] = interquartile
                    features[i,9,j] = skewness
                    features[i,10,j] = kurtosis
                    features[i,11,j] = acr
                    features[i,12,j] = meanFirstOrder
                    features[i,13,j] = meanNormFirstOrder
                    features[i,14,j] = meanSecondOrder
                    features[i,15,j] = meanNormSecondOrder
                    features[i,16,j] = spectralEnergy
                    features[i,17,j] = spectralEntropy
            np.save(resultDir+outputFileName, features)


def getZeroCrossings(mean, data):
    sign = [1,0] ;
    direction = 0
    countZC = 0;
    if data[0] >= mean:
        direction = 1

    for i in range(len(data)):
        if (data[i] >= mean and direction == 0) or (data[i] < mean and direction == 1):
            direction = sign[direction]
            countZC = countZC+1
    return countZC
    

#program starts:
generateAndSaveFeatures()
