import numpy as np
import os

inputDirForLabels = "/hdd/datasets/OPPORTUNITY/windows_labels/64/"
resultDir = "/hdd/opportunity/64/results/"                
NUM_OF_FEATURES = 18
NUM_OF_SENSORS = 107

def makeTrainingFeatureVector():
    dataS11 = np.load(resultDir+"S1-ADL1_features.npy")
    dataS12 = np.load(resultDir+"S1-ADL2_features.npy")
    dataS13 = np.load(resultDir+"S1-ADL3_features.npy")
    dataS14 = np.load(resultDir+"S1-ADL4_features.npy")
    dataS15 = np.load(resultDir+"S1-ADL5_features.npy")
    dataS1D = np.load(resultDir+"S1-Drill_features.npy")

    dataS21 = np.load(resultDir+"S2-ADL1_features.npy")
    dataS22 = np.load(resultDir+"S2-ADL2_features.npy")
    dataS23 = np.load(resultDir+"S2-ADL3_features.npy")
    dataS24 = np.load(resultDir+"S2-ADL4_features.npy")
    dataS25 = np.load(resultDir+"S2-ADL5_features.npy")
    dataS2D = np.load(resultDir+"S2-Drill_features.npy")

    dataS31 = np.load(resultDir+"S3-ADL1_features.npy")
    dataS32 = np.load(resultDir+"S3-ADL2_features.npy")
    dataS33 = np.load(resultDir+"S3-ADL3_features.npy")
    dataS34 = np.load(resultDir+"S3-ADL4_features.npy")
    dataS35 = np.load(resultDir+"S3-ADL5_features.npy")
    dataS3D = np.load(resultDir+"S3-Drill_features.npy")

    dataS41 = np.load(resultDir+"S4-ADL1_features.npy")
    dataS42 = np.load(resultDir+"S4-ADL2_features.npy")
    dataS43 = np.load(resultDir+"S4-ADL3_features.npy")
    dataS44 = np.load(resultDir+"S4-ADL4_features.npy")
    dataS45 = np.load(resultDir+"S4-ADL5_features.npy")
    dataS4D = np.load(resultDir+"S4-Drill_features.npy")

    dims = np.zeros((5, 7, 3))

    dims[1][1] =  np.shape(dataS11)
    dims[1][2] =  np.shape(dataS12)
    dims[1][3] =  np.shape(dataS13)
    dims[1][4] =  np.shape(dataS14)
    dims[1][5] =  np.shape(dataS15)
    dims[1][6] =  np.shape(dataS1D)

    dims[2][1] =  np.shape(dataS21)
    dims[2][2] =  np.shape(dataS22)
    dims[2][3] =  np.shape(dataS23)
    dims[2][4] =  np.shape(dataS24)
    dims[2][5] =  np.shape(dataS25)
    dims[2][6] =  np.shape(dataS2D)

    dims[3][1] =  np.shape(dataS31)
    dims[3][2] =  np.shape(dataS32)
    dims[3][3] =  np.shape(dataS33)
    dims[3][4] =  np.shape(dataS34)
    dims[3][5] =  np.shape(dataS35)
    dims[3][6] =  np.shape(dataS3D)

    dims[4][1] =  np.shape(dataS41)
    dims[4][2] =  np.shape(dataS42)
    dims[4][3] =  np.shape(dataS43)
    dims[4][4] =  np.shape(dataS44)
    dims[4][5] =  np.shape(dataS45)
    dims[4][6] =  np.shape(dataS4D)

    for i in range(5):
        for j in range(7):
            print(dims[i][j][0])

    print("*******")
    for i in range(7):
        print(dims[:,i,0])
        print("-------")
    print("*******")

    trainVecX = sum(dims[:,1,0] + dims[:,2,0] + dims[:,3,0] + dims[:,6,0])
    trainVecY = NUM_OF_FEATURES * NUM_OF_SENSORS

    #x = trainVecX[1] + trainVecX[2] + trainVecX[3] + trainVecX[4] 

    print(trainVecX, trainVecY)
    

    testVecX = dims[:,4,0] + dims[:,5,0]
    testVecY = trainVecY

    trainVec = np.zeros((int(trainVecX), trainVecY))  ## to store feature vectors
    print(np.shape(trainVec))

    position = 0
    

    print("creating training vector")
    print("S11...")
    data = dataS11
    for i in range(int(dims[1][1][0])):
        trainVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    print("S12...")
    print("position:", position)
    print("first value:",trainVec[0][0])
    print("last values:", trainVec[0][1925])
    print("first vector:", trainVec[0])
    print("last vector:", trainVec[1])
    print("length of a  vector:", np.shape(trainVec[0]))
    print("length of a  vector:", np.shape(trainVec[1]))
    
    
    data = dataS12
    for i in range(int(dims[1][2][0])):
        trainVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    print("S13...")
    data = dataS13
    for i in range(int(dims[1][3][0])):
        trainVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    print("S1D...")
    data = dataS1D
    for i in range(int(dims[1][6][0])):
        trainVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    print("S2X...")
    data = dataS21
    for i in range(int(dims[2][1][0])):
        trainVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    data = dataS22
    for i in range(int(dims[2][2][0])):
        trainVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    data = dataS23
    for i in range(int(dims[2][3][0])):
        trainVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    data = dataS2D
    for i in range(int(dims[2][6][0])):
        trainVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    print("S3X...")
    data = dataS31
    for i in range(int(dims[3][1][0])):
        trainVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    data = dataS32
    for i in range(int(dims[3][2][0])):
        trainVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    data = dataS33
    for i in range(int(dims[3][3][0])):
        trainVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    data = dataS3D
    for i in range(int(dims[3][6][0])):
        trainVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    print("S4X...")
    data = dataS41
    for i in range(int(dims[4][1][0])):
        trainVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    data = dataS42
    for i in range(int(dims[4][2][0])):
        trainVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    data = dataS43
    for i in range(int(dims[4][3][0])):
        trainVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    data = dataS4D
    for i in range(int(dims[4][6][0])):
        trainVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1

    print(np.shape(trainVec))
    np.save(resultDir+"training_features_matrix.npy", trainVec)
            



def makeTestingFeatureVector():
    dataS14 = np.load(resultDir+"S1-ADL4_features.npy")
    dataS15 = np.load(resultDir+"S1-ADL5_features.npy")

    dataS24 = np.load(resultDir+"S2-ADL4_features.npy")
    dataS25 = np.load(resultDir+"S2-ADL5_features.npy")

    dataS34 = np.load(resultDir+"S3-ADL4_features.npy")
    dataS35 = np.load(resultDir+"S3-ADL5_features.npy")

    dataS44 = np.load(resultDir+"S4-ADL4_features.npy")
    dataS45 = np.load(resultDir+"S4-ADL5_features.npy")

    dims = np.zeros((5, 7, 3))

    dims[1][4] =  np.shape(dataS14)
    dims[1][5] =  np.shape(dataS15)

    dims[2][4] =  np.shape(dataS24)
    dims[2][5] =  np.shape(dataS25)

    dims[3][4] =  np.shape(dataS34)
    dims[3][5] =  np.shape(dataS35)

    dims[4][4] =  np.shape(dataS44)
    dims[4][5] =  np.shape(dataS45)
    

    testVecX = sum(dims[:,4,0] + dims[:,5,0])
    testVecY = NUM_OF_FEATURES * NUM_OF_SENSORS


    testVec = np.zeros((int(testVecX), testVecY))  ## to store feature vectors
    print(np.shape(testVec))

    position = 0
    

    print("creating test vector")
    print("S14...")
    data = dataS14
    for i in range(int(dims[1][4][0])):
        testVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    
    print("position:", position)
    print("first value:",testVec[1][0])
    print("last values:", testVec[0][1925])
    print("first vector:", testVec[0])
    print("last vector:", testVec[1])
    print("length of a  vector:", np.shape(testVec[0]))
    print("length of a  vector:", np.shape(testVec[1]))
    
    print("S15...")
    data = dataS15
    for i in range(int(dims[1][5][0])):
        testVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1

    print("S24...")
    data = dataS24
    for i in range(int(dims[2][4][0])):
        testVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1

    print("S25...")
    data = dataS25
    for i in range(int(dims[2][5][0])):
        testVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1
    print("S34...")
    data = dataS34
    for i in range(int(dims[3][4][0])):
        testVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1

    print("S35...")
    data = dataS35
    for i in range(int(dims[3][5][0])):
        testVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1

    print("S44...")
    data = dataS44
    for i in range(int(dims[4][4][0])):
        testVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1

    print("S45...")
    data = dataS45
    for i in range(int(dims[4][5][0])):
        testVec[position] = np.reshape(data[i], (1,np.product(data[i].shape)))[0]
        position = position + 1



    

    print(np.shape(testVec))
    np.save(resultDir+"testing_features_matrix.npy", testVec)
    

def makeTrainingTestingLabelVectors():
    trainVec_labels = np.array([])
    testVec_labels = np.array([])
    print("Integrating training labels...")
    #integrating training labels 
    for sNum in range(1,5):
        for num in range(1,4):
            cFileName = 'S'+str(sNum)+'-ADL'+str(num)+'_labels.npy' # complete file name
            #data = np.load('training_labels/'+cFileName)
            data = np.load(inputDirForLabels+cFileName)
            print('Processing file: ' + cFileName, np.shape(data))
            trainVec_labels = np.append(trainVec_labels,data)
        dFileName ='S'+str(sNum)+'-Drill_labels.npy' # complete file name
        #data = np.load('training_labels/'+dFileName)
        data = np.load(inputDirForLabels+dFileName)
        print('Processing file: ' + dFileName, np.shape(data))
        trainVec_labels = np.append(trainVec_labels,data)

    print(np.shape(trainVec_labels))


    #integrating testing labels 
    print("Integrating testing labels...")
    for sNum in range(1,5):
        for num in range(4,6):
            cFileName = 'S'+str(sNum)+'-ADL'+str(num)+'_labels.npy' # complete file name 
            #data = np.load('testing_labels/'+cFileName)
            data = np.load(inputDirForLabels+cFileName)
            print('Processing file: ' + cFileName, np.shape(data))
            testVec_labels = np.append(testVec_labels,data)

    print(np.shape(testVec_labels))

    np.save(resultDir+"training_labels.npy", trainVec_labels)
    np.save(resultDir+"testing_labels.npy", testVec_labels)


        
    
makeTrainingFeatureVector()
makeTestingFeatureVector()
makeTrainingTestingLabelVectors()              

