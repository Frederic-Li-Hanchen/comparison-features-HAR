import keras
from keras import backend as K

import numpy as np
import os
import sys
import time
from utilitary import *
from dnnModels import *
from trainDnn import modelParamPath, dataPath, dnnFeaturesPath, selectedModel, batchSize


#-------------------------------------------------------------------------------------------------------
# Function extractDnnFeatures: compute the features on the data and save them at the location
# specified by savePath
#-------------------------------------------------------------------------------------------------------
def extractDnnFeatures(
	modelPath,
    dataPath,
	savePath,
    nbClasses,
    selectedModel,
    batchSize
	):

  start = time.time()

  # Load the data and labels
  print('Loading the training and testing data ...')

  x_train,y_train,trainingShape, x_test,y_test,testingShape,_,_ = loadCleanOpportunityData(dataPath,permute=False)
  assert trainingShape[1] == testingShape[1] # Window size
  assert trainingShape[2] == testingShape[2] # Nb of sensors

  if selectedModel.name == 'AE':
    x_train = x_train.reshape(trainingShape[0],trainingShape[1]*trainingShape[2],)
    input_shape = (trainingShape[1]*trainingShape[2],)
    x_test = x_test.reshape(testingShape[0],testingShape[1]*testingShape[2],)
    input_shape = (testingShape[1]*testingShape[2], )
  else:
    x_train = x_train.reshape(trainingShape[0],trainingShape[1],trainingShape[2], 1)
    input_shape = (trainingShape[1], trainingShape[2], 1)
    x_test = x_test.reshape(testingShape[0],testingShape[1],testingShape[2], 1)
    input_shape = (testingShape[1], testingShape[2], 1)
      
  nbTrainingExamples = trainingShape[0]
  assert len(y_train) == nbTrainingExamples
  print('   %d training examples loaded' % (nbTrainingExamples))
  nbTestingExamples = testingShape[0]
  assert len(y_test) == nbTestingExamples
  print('   %d testing examples loaded' % (nbTestingExamples))

  ### Model definition
  print('Building the model ...')
  selectedModel.getModelName()
  params = selectedModel.params

  if selectedModel.name == 'MLP':
    if params['nb_dense_layers'] == 1:
      model = normMlp3(inputShape=input_shape,
                       inputMLP=params['dense_size'][0],
                       activationMLP=params['activation'],
                       nbClasses=nbClasses,
                       withSoftmax=False)
    elif params['nb_dense_layers'] == 2:
      model = normMlp3(inputShape=input_shape,
                       inputMLP1=params['dense_size'][0],
                       inputMLP2=params['dense_size'][1],
                       activationMLP=params['activation'],
                       nbClasses=nbClasses,
                       withSoftmax=False)
    elif params['nb_dense_layers'] == 3:
      model = normMlp3(inputShape=input_shape,
                       inputMLP1=params['dense_size'][0],
                       inputMLP2=params['dense_size'][1],
                       inputMLP3=params['dense_size'][2],
                       activationMLP=params['activation'],
                       nbClasses=nbClasses,
                       withSoftmax=False)
    else:
      print('-----------------------------------------------------------------------------')
      print('ERROR: Only MLP models with 1, 2 or 3 dense layers are currently implemented!')
      print('-----------------------------------------------------------------------------')
      sys.exit()

  elif selectedModel.name == 'CNN':
    if params['nb_conv_blocks'] == 1:
      model = normConv1(inputShape=input_shape,
                        nkerns=params['nb_conv_kernels'],
                        filterSizes=params['conv_kernels_size'],
                        poolSizes=params['pooling_size'],
                        activationConv=params['conv_activation'],
                        inputMLP=params['dense_size'][0],
                        activationMLP=params['dense_activation'],
                        nbClasses=nbClasses,
                        withSoftmax=False)

    elif params['nb_conv_blocks'] == 2:
      model = normConv2(inputShape=input_shape,
                        nkerns=params['nb_conv_kernels'],
                        filterSizes=params['conv_kernels_size'],
                        poolSizes=params['pooling_size'],
                        activationConv=params['conv_activation'],
                        inputMLP=params['dense_size'][0],
                        activationMLP=params['dense_activation'],
                        nbClasses=nbClasses,
                        withSoftmax=False)

    elif params['nb_conv_blocks'] == 3:
      model = normConv3(inputShape=input_shape,
                        nkerns=params['nb_conv_kernels'],
                        filterSizes=params['conv_kernels_size'],
                        poolSizes=params['pooling_size'],
                        activationConv=params['conv_activation'],
                        inputMLP=params['dense_size'][0],
                        activationMLP=params['dense_activation'],
                        nbClasses=nbClasses,
                        withSoftmax=False)
    else:
      print('-------------------------------------------------------------------------------------')
      print('ERROR: Only CNN models with 1, 2 or 3 convolutional blocks are currently implemented!')
      print('-------------------------------------------------------------------------------------')
      sys.exit()

  elif selectedModel.name == 'LSTM':
    if params['nb_lstm_layers'] == 1:
      model = normLstm(inputShape=input_shape,
                       outputLSTM=params['lstm_output_dim'][0],
                       inputMLP=params['dense_size'][0],
                       activationMLP=params['dense_activation'],
                       nbClasses=nbClasses,
                       withSoftmax=False)
    elif params['nb_lstm_layers'] == 2:
      model = normLstm2(inputShape=input_shape,
                        outputLSTM1=params['lstm_output_dim'][0],
                        outputLSTM2=params['lstm_output_dim'][1],
                        inputMLP=params['dense_size'][0],
                        activationMLP=params['dense_activation'],
                        nbClasses=nbClasses,
                        withSoftmax=False)
    else:
      print('---------------------------------------------------------------------')
      print('ERROR: Only LSTM models with 1 or 2 layers are currently implemented!')
      print('---------------------------------------------------------------------')
      sys.exit()

  elif selectedModel.name == 'Hybrid':
    if (params['nb_conv_blocks'], params['nb_lstm_layers']) == (1,2):
      model = normConv1Lstm2(inputShape=input_shape,
                             nkerns=params['nb_conv_kernels'],
                             filterSizes=params['conv_kernels_size'],
                             poolSizes=params['pooling_size'],
                             activationConv=params['conv_activation'],
                             outputLSTM1=params['lstm_output_dim'][0],
                             outputLSTM2=params['lstm_output_dim'][1],
                             inputMLP=params['dense_size'][0],
                             activationMLP=params['dense_activation'],
                             nbClasses=nbClasses,
                             withSoftmax=False)

    else:
      print('------------------------------------------------------------------------------------------------')
      print('ERROR: Only Hybrid models with 1 convolutional block and 2 LSTM layers are currently implemented!')
      print('------------------------------------------------------------------------------------------------')
      sys.exit()

  elif selectedModel.name == 'AE':
    if params['nb_layers_encoder'] == 1:
      model = autoencoder1(inputShape=input_shape,
                           inputMLP=params['dense_size'][0],
                           activationMLP=params['dense_activation'],
                           decoder=False)  
    elif params['nb_layers_encoder'] == 2:
      model = autoencoder2(inputShape=input_shape,
                           inputMLP1=params['dense_size'][0],
                           inputMLP2=params['dense_size'][1],
                           activationMLP=params['dense_activation'],
                           decoder=False)
    elif params['nb_layers_encoder'] == 3:
      model = autoencoder2(inputShape=input_shape,
                           inputMLP1=params['dense_size'][0],
                           inputMLP2=params['dense_size'][1],
                           inputMLP3=params['dense_size'][2],
                           activationMLP=params['dense_activation'],
                           decoder=False)
    else:
      print('-------------------------------------------------------------------------------')
      print('ERROR: Only AE models with 1, 2 or 3 encoder layers are currently implemented!')
      print('-------------------------------------------------------------------------------')
      sys.exit()

  else:
    print('--------------------------------------------------------------------------------')
    print('ERROR: Incorrect model name! Currently supported: MLP, CNN, LSTM, Hybrid and AE')
    print('--------------------------------------------------------------------------------')
    sys.exit()

  # Load the save weights
  print('Loading pre-learned weights ...')
  model.load_weights(modelPath,by_name=True)

  # Print a model summary
  model.summary()

  # Allocate the feature arrays
  featureSize = params['dense_size'][-1]
  trainingDnnFeatures = np.empty((nbTrainingExamples,featureSize),dtype=np.float32)
  testingDnnFeatures = np.empty((nbTestingExamples,featureSize),dtype=np.float32)

  # Compute the DNN features
  print('Computing DNN features on the training set...')
  idx = 0
  while idx < nbTrainingExamples:
    if idx + batchSize < nbTrainingExamples:
      endIdx = idx+batchSize
      size = batchSize
    else:
      endIdx = nbTrainingExamples
      size = nbTrainingExamples-idx
    predictions = model.predict(x_train[idx:endIdx],batch_size=size)
    trainingDnnFeatures[idx:endIdx] = predictions
    idx += batchSize

  print('Computing DNN features on the testing set...')
  idx = 0
  while idx < nbTestingExamples:
    if idx + batchSize < nbTestingExamples:
      endIdx = idx+batchSize
      size = batchSize
    else:
      endIdx = nbTestingExamples
      size = nbTestingExamples-idx
    predictions = model.predict(x_test[idx:endIdx],batch_size=size)
    testingDnnFeatures[idx:endIdx] = predictions
    idx += batchSize

  # Save features and labels
  print('Saving results ...')

  np.save(savePath+'/dnnFeatures_training.npy',trainingDnnFeatures)
  np.save(savePath+'/dnnLabels_training.npy',y_train)
  np.save(savePath+'/dnnFeatures_testing.npy',testingDnnFeatures)
  np.save(savePath+'/dnnLabels_testing.npy',y_test)

  end = time.time()

  print('Results saved in folder %s' % (savePath))
  print('Script ran for %.2f seconds' % (end-start))



#-------------------------------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

  print('#############################################################')
  print('Starting DNN features computation ...')
  print('#############################################################')

  # Computation of features on the training and testing sets using the trained model
  extractDnnFeatures(modelPath=modelParamPath+'/model.h5',
                     dataPath=dataPath,
                     savePath=dnnFeaturesPath,
                     nbClasses=nbClasses,
                     selectedModel=selectedModel,
                     batchSize=batchSize)