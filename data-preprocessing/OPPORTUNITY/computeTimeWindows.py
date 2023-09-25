#-------------------------------------------------------------------------------------------------------
# Script to extract the time windows of data + labels from the clean OPPORTUNITY data files
#
# Inputs: 
#   - [string] dataFolder: path to the folder containing the cleaned .txt data files
#   - [string] resultFolder: path to the folder to save the results
#   - [int] timeWindow: size of the time window
#   - [int] slidingStride: stride of the time window
#
# This script saves the time windows in a .npy file containing a 3D array of size 
# nb_windows x time_window_size x nb_sensors
# The labels are also saved in a .npy file containing a vector with the corresponding labels
# Note: one data and label .npy files are created for each .txt file in the data folder
#-------------------------------------------------------------------------------------------------------


import numpy as np
import os
import sys
from math import floor


#-------------------------------------------------------------------------------------------------------
# Hyper-parameters -> TODO: change and check before use
#-------------------------------------------------------------------------------------------------------

# Time window parameters
timeWindow = 64
slidingStride = 3

# Number of sensors ranked by variance to use
nbSensorsByVar = 107

# dataFolder: path to the folder containing the clean OPPORTUNITY data
# resultFolder: path to the folder where to save the data frames
if nbSensorsByVar == 107:
    dataFolder = '/hdd/datasets/OPPORTUNITY/clean/all_sensors/'
    resultFolder = '/hdd/datasets/OPPORTUNITY/windows_labels/all_sensors/'
else:
    dataFolder = '/hdd/datasets/OPPORTUNITY/clean/'+str(nbSensorsByVar)+'_highest_var_sensors/'
    resultFolder = '/hdd/datasets/OPPORTUNITY/windows_labels/'+str(nbSensorsByVar)+'_highest_var_sensors/'



#-------------------------------------------------------------------------------------------------------
# Function extractTimeWindowsAndLabels: load the OPPORTUNITY data, and extract and save the time windows 
# and labels as .npy files
#-------------------------------------------------------------------------------------------------------

def extractTimeWindowsAndLabels(pathToDataFolder=dataFolder,resultFolder=resultFolder,timeWindow=timeWindow,slidingStride=slidingStride): 

    # List files in the data folder
    print('Input data file folder: %s' % (pathToDataFolder))
    dataFileList = os.listdir(pathToDataFolder)

    # Extraction of the time windows + labels for each .txt data file
    for fileName in dataFileList:

        if 'ADL' in fileName or 'Drill' in fileName:

            print('Processing file %s ...' % (fileName))

            # Get file contents as string
            fh = open(pathToDataFolder+'/'+fileName,'r')
            contents = fh.readlines()
            fh.close()

            # Convert to a matrix of floats
            # Note: the gesture label is in the last column. 
            nbTimestamps = len(contents)
            nbSensors = len([np.float32(e) for e in contents[0].split()])-1
            data = np.zeros((nbTimestamps,nbSensors))
            labels = -1*np.ones(nbTimestamps,dtype=int) 

            for idx in range(nbTimestamps):
                dataLineTmp = [np.float32(e) for e in contents[idx].split()]
                dataLine = dataLineTmp[:-1] 
                data[idx] = dataLine
                labels[idx] = int(dataLineTmp[-1])

            # Determination of the total number of time windows, and pre-allocation of result arrays
            nbTimeWindows = int(floor((nbTimestamps-timeWindow)/slidingStride))+1
            timeWindowArray = np.empty((nbTimeWindows,timeWindow,nbSensors),dtype=np.float32)
            labelsVector = -1*np.ones((nbTimeWindows),dtype=int)

            # Iteration on the data file to build the examples of size timeWindow x nbOfSensors
            idx = 0
            timeWindowCounter = 0
            while idx < nbTimestamps - timeWindow + 1:
                windowData = data[idx:idx+timeWindow]
                windowLabels = labels[idx:idx+timeWindow]
                # Determine the majoritary label among those of the timeWindow examples considered | TODO: uncomment for the majority label approach
                (values,counts) = np.unique(windowLabels,return_counts=True)
                majoritaryLabel = values[np.argmax(counts)] 
                #majoritaryLabel = windowLabels[-1] # TODO: uncomment for the last label solution
                # Store the data and labels
                timeWindowArray[timeWindowCounter] = windowData
                labelsVector[timeWindowCounter] = majoritaryLabel
                # Iterate
                timeWindowCounter += 1
                idx += slidingStride 

            # Project labels in the range (0,L-1) with L number of different labels
            unique = set(windowLabels)
            cleanLabels = [list(unique).index(l) for l in windowLabels]

            # Save data and labels in the result folder as .npy files 
            resultname = fileName.replace('.txt','')
            np.save(resultFolder+'/'+resultname+'_data.npy',timeWindowArray)
            np.save(resultFolder+'/'+resultname+'_labels.npy',labelsVector)
        
    print('Results saved in the folder %s' % (resultFolder))


#-------------------------------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    extractTimeWindowsAndLabels()
