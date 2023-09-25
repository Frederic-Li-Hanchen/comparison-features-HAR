import keras
from keras.optimizers import RMSprop
from keras import backend as K

import numpy as np
import os
import sys
import time

from utilitary import *
from dnnModels import *

##############################################################################################################
#-------------------------------------------------------------------------------------------------------
# Hyper-parameters ---> TODO: modify this part of the script according to your needs
#-------------------------------------------------------------------------------------------------------
### Input data parameters
nbSensors = 107 # Options: 5, 10, 20, 50, 80, 107
timeWindow = 64 # Time window, "width" of the "input image". Options: 32, 64, 96 if nbSensors == 107. Otherwise, timeWindow == 64.

### Data paths
if nbSensors == 107:
    pathExtension = 'all_sensors/'+str(timeWindow)
else:
    pathExtension = str(nbSensors)+'_highest_var_sensors/' 

dataPath = '/hdd/datasets/OPPORTUNITY/windows_labels/'+pathExtension # Path to the data
modelParamPath = '/home/prg/programming/python/OPPORTUNITY/model' # Path to save the trained DNN model
dnnFeaturesPath = '/home/prg/programming/python/OPPORTUNITY/dnn_features' # Path to save the DNN features computed with the trained model

### Training parameters
batchSize = 500
nbEpochs = 50
learningRate = 0.05 # NOTE: irrelevant if the ADADELTA optimization is used 

### Model parameters
mlp = modelParam(
  'MLP',
  {
   "nb_dense_layers" : 3,
   "dense_size" : [2000,2000,2000],
   "activation" : 'relu',
  }
)

cnn = modelParam(
  'CNN',
  {
   "nb_conv_blocks" : 3,
   "nb_conv_kernels" : [50,40,30],
   "conv_kernels_size" : [(11,1),(10,1),(6,1)],
   "pooling_size" : [(2,1),(3,1),(1,1)],
   "conv_activation" : 'relu',
   "dense_size" : [1000],
   "dense_activation" : 'relu',
  }
)

lstm = modelParam(
  'LSTM',
  {
   "nb_lstm_layers": 2,
   "lstm_output_dim" : [600,600],
   "dense_size" : [512],
   "dense_activation" : 'relu',
  }
)

hybrid = modelParam(
  'Hybrid',
  {
   "nb_conv_blocks" : 1,
   "nb_conv_kernels" : [50],
   "conv_kernels_size" : [(11,1)],
   "pooling_size" : [(2,1)],
   "conv_activation" : 'relu',
   "nb_lstm_layers": 2,
   "lstm_output_dim" : [600,600],
   "dense_size" : [512],
   "dense_activation" : 'relu',
  }
)

ae = modelParam(
  'AE',
  {
   "nb_layers_encoder" : 1,
   "dense_size" : [5000],
   "dense_activation" : 'relu',
  }
)

selectedModel = mlp # TODO: modify this value to change the model. Choices available: mlp, cnn, lstm, hybrid, ae 

### SVM soft-margin parameter | NOTE: if several values are provided, several SVM will be trained and evaluated
#C = [2**e for e in range(-6,6)]
C = [2**(-6)]
##############################################################################################################



#-------------------------------------------------------------------------------------------
# Function trainDnn: train a Deep Neural Network on the OPPORTUNITY dataset and save the model weights 
# at the location specified by modelPath 
#-------------------------------------------------------------------------------------------------------
def trainDnn(dataPath,
             modelPath,
             learningRate,
             epochs, 
             batchSize,
             nbClasses, 
             windowSize, 
             nbSensors,
             selectedModel
            ):

  start = time.time()

  ### load the OPPORTUNITY data
  print('Loading the OPPORTUNITY data ...')

  x_train, y_train, trainShape, x_test, y_test, testShape, _, _ = loadCleanOpportunityData(dataPath,permute=True)
  assert trainShape[1] == testShape[1] # Size of the time window
  assert trainShape[2] == testShape[2] # Number of sensors

  if selectedModel.name == 'AE':
    x_train = x_train.reshape(trainShape[0],trainShape[1]*trainShape[2],)
    x_test = x_test.reshape(testShape[0],testShape[1]*testShape[2],)
    input_shape = (testShape[1]* testShape[2],)
    print('x_train shape: (%d, %d)' % (trainShape[0], trainShape[1]*trainShape[2]))
    
  else:
    x_train = x_train.reshape(trainShape[0],trainShape[1],trainShape[2], 1)
    x_test = x_test.reshape(testShape[0],testShape[1],testShape[2], 1)
    input_shape = (testShape[1], testShape[2], 1)
    print('x_train shape: (%d, %d, %d)' % (trainShape[0], trainShape[1], trainShape[2]))


  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  print('%d train samples' % (trainShape[0]))
  print('%d test samples' % (testShape[0]))

  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, nbClasses)
  y_test = keras.utils.to_categorical(y_test, nbClasses)

  ### Model definition
  print('Building the model ...')
  selectedModel.getModelName()
  params = selectedModel.params

  if selectedModel.name == 'MLP':
    if params['nb_dense_layers'] == 1:
      model = normMlp1(inputShape=input_shape,
                       inputMLP=params['dense_size'][0],
                       activationMLP=params['activation'],
                       nbClasses=nbClasses)
    elif params['nb_dense_layers'] == 2:
      model = normMlp2(inputShape=input_shape,
                       inputMLP1=params['dense_size'][0],
                       inputMLP2=params['dense_size'][1],
                       activationMLP=params['activation'],
                       nbClasses=nbClasses)
    elif params['nb_dense_layers'] == 3:
      model = normMlp3(inputShape=input_shape,
                       inputMLP1=params['dense_size'][0],
                       inputMLP2=params['dense_size'][1],
                       inputMLP3=params['dense_size'][2],
                       activationMLP=params['activation'],
                       nbClasses=nbClasses)
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
                        nbClasses=nbClasses)

    elif params['nb_conv_blocks'] == 2:
      model = normConv2(inputShape=input_shape,
                        nkerns=params['nb_conv_kernels'],
                        filterSizes=params['conv_kernels_size'],
                        poolSizes=params['pooling_size'],
                        activationConv=params['conv_activation'],
                        inputMLP=params['dense_size'][0],
                        activationMLP=params['dense_activation'],
                        nbClasses=nbClasses)

    elif params['nb_conv_blocks'] == 3:
      model = normConv3(inputShape=input_shape,
                        nkerns=params['nb_conv_kernels'],
                        filterSizes=params['conv_kernels_size'],
                        poolSizes=params['pooling_size'],
                        activationConv=params['conv_activation'],
                        inputMLP=params['dense_size'][0],
                        activationMLP=params['dense_activation'],
                        nbClasses=nbClasses)
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
                       nbClasses=nbClasses)
    elif params['nb_lstm_layers'] == 2:
      model = normLstm2(inputShape=input_shape,
                        outputLSTM1=params['lstm_output_dim'][0],
                        outputLSTM2=params['lstm_output_dim'][1],
                        inputMLP=params['dense_size'][0],
                        activationMLP=params['dense_activation'],
                        nbClasses=nbClasses)
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
                             nbClasses=nbClasses)

    else:
      print('------------------------------------------------------------------------------------------------')
      print('ERROR: Only Hybrid models with 1 convolutional block and 2 LSTM layers are currently implemented!')
      print('------------------------------------------------------------------------------------------------')
      sys.exit()

  elif selectedModel.name == 'AE':
    if params['nb_layers_encoder'] == 1:
      model = autoencoder1(inputShape=input_shape,
                           inputMLP=params['dense_size'][0],
                           activationMLP=params['dense_activation'])  
    elif params['nb_layers_encoder'] == 2:
      model = autoencoder2(inputShape=input_shape,
                           inputMLP1=params['dense_size'][0],
                           inputMLP2=params['dense_size'][1],
                           activationMLP=params['dense_activation'])
    elif params['nb_layers_encoder'] == 3:
      model = autoencoder3(inputShape=input_shape,
                           inputMLP1=params['dense_size'][0],
                           inputMLP2=params['dense_size'][1],
                           inputMLP3=params['dense_size'][2],
                           activationMLP=params['dense_activation'])
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

  # Model compilation:
  if selectedModel.name == 'AE':   
    model.compile(loss='mse', # For autoencoders
                  optimizer=keras.optimizers.Adadelta(),
                  #optimizer=keras.optimizers.sgd(lr=learningRate,decay=1e-6),
                  metrics=['acc', fmeasure])
  else:
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  #optimizer=keras.optimizers.adagrad(lr=learningRate),
                  metrics=['acc', fmeasure])

  # Print a model summary
  model.summary()

  # Tensorboard report
  if not os.path.exists('/tmp/logs/'):
    os.makedirs('/tmp/logs/')
  logFilesList= os.listdir('/tmp/logs/')
  if logFilesList != []:
    for file in logFilesList:
      os.remove('/tmp/logs/'+file)
  if not os.path.exists('/tmp/checkpoint/'):
    os.makedirs('/tmp/checkpoint/')
  tensorboard = keras.callbacks.TensorBoard(log_dir='/tmp/logs/', histogram_freq=0, write_graph=True, write_images=True)
  checkpoint = keras.callbacks.ModelCheckpoint('/tmp/checkpoint/checkpoint.hdf5',monitor='fmeasure',save_best_only=True,save_weights_only=False)

  print('Initiating the training phase ...')

  
  if selectedModel.name == 'AE':
    # Autoencoder unsupervised training
    model.fit(x_train, x_train,
              batch_size=batchSize,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, x_test),
              callbacks=[tensorboard,checkpoint])
    model.load_weights('/tmp/checkpoint/checkpoint.hdf5')
    print('Model evaluation')
    score = model.evaluate(x_test, x_test, verbose=0)
    
  else:
    # Supervised training
    model.fit(x_train, y_train,
              batch_size=batchSize,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard,checkpoint])
    model.load_weights('/tmp/checkpoint/checkpoint.hdf5')
    print('Model evaluation')
    score = model.evaluate(x_test, y_test, verbose=0)

  end = time.time()

  print('##############################################')
  print('Total time used: %.2f seconds' % (end-start))
  print('Tensorboard log file generated in the directory /tmp/logs/')
  print('Use the command')
  print('    tensorboard --logdir /tmp/logs/')
  print('to read it')
  for idx in range(len(score)):
    print('%s: %.3f' % (model.metrics_names[idx],score[idx]))
  print('##############################################')

  # Save the weights of the network
  model_json = model.to_json()
  with open(modelPath + "/model.json", "w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights(modelPath + "/model.h5")
  print("Saved model to folder:" + modelPath)


#-------------------------------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
  print('#############################################################')
  print('Starting the training of the DNN model ...')
  print('#############################################################')
  trainDnn(dataPath=dataPath,
           modelPath=modelParamPath,
           learningRate=learningRate,
           epochs=nbEpochs, 
           batchSize=batchSize,
           nbClasses=nbClasses, 
           windowSize=timeWindow, 
           nbSensors=nbSensors,
           selectedModel=selectedModel)
