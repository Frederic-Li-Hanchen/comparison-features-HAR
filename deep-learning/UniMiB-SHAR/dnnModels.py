from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Merge, Add, merge
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional



#-------------------------------------------------------------------------------------------------------
# normMlp1: define a batch normalization + one hidden layer MLP
#-------------------------------------------------------------------------------------------------------
def normMlp1(
    inputShape,
    inputMLP,
    activationMLP,
    nbClasses,
    withSoftmax=True):

    model = Sequential()

    model.add(BatchNormalization(input_shape=inputShape))

    model.add(Flatten())
    model.add(Dense(inputMLP, activation=activationMLP))

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    # Return the model
    return model


#-------------------------------------------------------------------------------------------------------
# normMlp2: define a batch normalization + 2 hidden layers MLP
#-------------------------------------------------------------------------------------------------------
def normMlp2(
    inputShape,
    inputMLP1,
    inputMLP2,
    activationMLP,
    nbClasses,
    withSoftmax=True):

    model = Sequential()

    model.add(BatchNormalization(input_shape=inputShape))

    model.add(Flatten())
    model.add(Dense(inputMLP1, activation=activationMLP))
    model.add(Dense(inputMLP2, activation=activationMLP))

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    # Return the model
    return model


#-------------------------------------------------------------------------------------------------------
# normMlp3: define a batch normalization + 3 hidden layers MLP
#-------------------------------------------------------------------------------------------------------
def normMlp3(
    inputShape,
    inputMLP1,
    inputMLP2,
    inputMLP3,
    activationMLP,
    nbClasses,
    withSoftmax=True):

    model = Sequential()

    model.add(BatchNormalization(input_shape=inputShape))
    model.add(Flatten())
    model.add(Dense(inputMLP1, activation=activationMLP))
    model.add(Dense(inputMLP2, activation=activationMLP))
    model.add(Dense(inputMLP3, activation=activationMLP))

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    # Return the model
    return model



#-------------------------------------------------------------------------------------------------------
# normConv1: define the batch normalization + 1 convolutional/max-pooling DNN
#-------------------------------------------------------------------------------------------------------
def normConv1(
    inputShape,
    nkerns,
    filterSizes,
    poolSizes,
    activationConv,
    inputMLP,
    activationMLP,
    nbClasses,
    withSoftmax=True):

    model = Sequential()

    model.add(BatchNormalization(input_shape=inputShape))

    # 1st convolutional + pooling + normalization layer 
    model.add(Conv2D(nkerns[0], kernel_size=filterSizes[0], activation=activationConv))
    model.add(MaxPooling2D(pool_size=poolSizes[0]))

    # Fully-connected layer
    model.add(Flatten())
    model.add(Dense(inputMLP, activation=activationMLP))

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    # Return the model
    return model

#-------------------------------------------------------------------------------------------------------
# normConv2: define the batch normalization + 2 convolutional/max-pooling DNN
#-------------------------------------------------------------------------------------------------------
def normConv2(
    inputShape,
    nkerns,
    filterSizes,
    poolSizes,
    activationConv,
    inputMLP,
    activationMLP,
    nbClasses,
    withSoftmax=True):

    model = Sequential()

    model.add(BatchNormalization(input_shape=inputShape))

    # 1st convolutional + pooling + normalization layer 
    model.add(Conv2D(nkerns[0], kernel_size=filterSizes[0], activation=activationConv))
    model.add(MaxPooling2D(pool_size=poolSizes[0]))

    # 2nd convolutional + pooling + normalization layer
    model.add(Conv2D(nkerns[1], kernel_size=filterSizes[1], activation=activationConv))
    model.add(MaxPooling2D(pool_size=poolSizes[1]))
  
    # Fully-connected layer
    model.add(Flatten())
    model.add(Dense(inputMLP, activation=activationMLP))
    #model.add(Dropout(0.5))

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    # Return the model
    return model


#-------------------------------------------------------------------------------------------------------
# normConv3: define the batch normalization + 3 convolutional/max-pooling DNN
#-------------------------------------------------------------------------------------------------------
def normConv3(
    inputShape,
    nkerns,
    filterSizes,
    poolSizes,
    activationConv,
    inputMLP,
    activationMLP,
    nbClasses,
    withSoftmax=True):

    model = Sequential()

    model.add(BatchNormalization(input_shape=inputShape))

    # 1st convolutional + pooling + normalization layer 
    #model.add(Conv2D(nkerns[0], kernel_size=filterSizes[0], activation='linear', input_shape=input_shape))
    model.add(Conv2D(nkerns[0], kernel_size=filterSizes[0], activation=activationConv))
    model.add(MaxPooling2D(pool_size=poolSizes[0]))

    # 2nd convolutional + pooling + normalization layer
    model.add(Conv2D(nkerns[1], kernel_size=filterSizes[1], activation=activationConv))
    model.add(MaxPooling2D(pool_size=poolSizes[1]))

    # 3rd block: convolutional + RELU + normalization
    model.add(Conv2D(nkerns[2], kernel_size=filterSizes[2], activation=activationConv))
    model.add(MaxPooling2D(pool_size=poolSizes[2]))
  
    # Fully-connected layer
    model.add(Flatten())
    model.add(Dense(inputMLP, activation=activationMLP))

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    # Return the model
    return model

#-------------------------------------------------------------------------------------------------------
# normLstm: define a batch normalization + LSTM DNN
#-------------------------------------------------------------------------------------------------------
def normLstm(
    inputShape,
    outputLSTM,
    inputMLP,
    activationMLP,
    nbClasses,
    withSoftmax=True):

    model = Sequential()

    # Batch normalization layer
    model.add(BatchNormalization(input_shape=inputShape))

    # LSTM layer with a many-to-one implementation
    # Note: size of the output = (outputSizeLastConv, outputLSTM)
    model.add(Reshape((inputShape[0],inputShape[1])))
    model.add(LSTM(outputLSTM))
  
    # Fully-connected layer
    model.add(Dense(inputMLP, activation=activationMLP))

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    # Return the model
    return model


#-------------------------------------------------------------------------------------------------------
# normLstm2: define a batch normalization + 2 LSTM DNN
#-------------------------------------------------------------------------------------------------------
def normLstm2(
    inputShape,
    outputLSTM1,
    outputLSTM2,
    inputMLP,
    activationMLP,
    nbClasses,
    withSoftmax=True):

    model = Sequential()

    # Batch normalization layer
    model.add(BatchNormalization(input_shape=inputShape))

    # LSTM layer with a many-to-one implementation
    # Note: size of the output = (outputSizeLastConv, outputLSTM)
    model.add(Reshape((inputShape[0],inputShape[1])))
    model.add(LSTM(outputLSTM1,return_sequences=True))
    model.add(LSTM(outputLSTM2,return_sequences=False))
  
    # Fully-connected layer
    model.add(Dense(inputMLP, activation=activationMLP))

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    # Return the model
    return model



#-------------------------------------------------------------------------------------------------------
# normConv1Lstm: define a batch normalization + 2 convolutional/max-pooling + LSTM DNN
#-------------------------------------------------------------------------------------------------------
def normConv1Lstm(
    inputShape,
    nkerns,
    filterSizes,
    poolSizes,
    activationConv,
    outputLSTM,
    inputMLP,
    activationMLP,
    nbClasses,
    withSoftmax=True):

    outputSizeLastConv = (inputShape[0]-filterSizes[0][0]+1)/poolSizes[0][0]

    model = Sequential()

    # Batch normalization layer
    model.add(BatchNormalization(input_shape=inputShape))

    # Convolution + max-pooling layers
    model.add(Conv2D(nkerns[0], kernel_size=filterSizes[0], activation=activationConv))
    model.add(MaxPooling2D(pool_size=poolSizes[0]))

    # LSTM layer with a many-to-one implementation
    # Note: size of the output = (outputSizeLastConv, outputLSTM)
    model.add(Reshape((outputSizeLastConv,inputShape[1]*nkerns[0]))) 
    model.add(LSTM(outputLSTM,return_sequences=False))

    # Fully-connected layer
    model.add(Dense(inputMLP, activation=activationMLP))

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    # Print the summary of the model
    #model.summary

    # Return the model
    return model



#-------------------------------------------------------------------------------------------------------
# normConv1Lstm: define a batch normalization + 2 convolutional/max-pooling + LSTM DNN
#-------------------------------------------------------------------------------------------------------
def normConv1Lstm2(
    inputShape,
    nkerns,
    filterSizes,
    poolSizes,
    activationConv,
    outputLSTM1,
    outputLSTM2,
    inputMLP,
    activationMLP,
    nbClasses,
    withSoftmax=True):

    outputSizeLastConv = (inputShape[0]-filterSizes[0][0]+1)/poolSizes[0][0]

    model = Sequential()

    # Batch normalization layer
    model.add(BatchNormalization(input_shape=inputShape))

    # Convolution + max-pooling layers
    model.add(Conv2D(nkerns[0], kernel_size=filterSizes[0], activation=activationConv))
    model.add(MaxPooling2D(pool_size=poolSizes[0]))

    # LSTM layer with a many-to-many implementation
    # Note: size of the output = (outputSizeLastConv, outputLSTM)
    model.add(Reshape((outputSizeLastConv,inputShape[1]*nkerns[0]))) 
    model.add(LSTM(outputLSTM1,return_sequences=True))

    # LSTM layer with a many-to-one implementation
    model.add(LSTM(outputLSTM2,return_sequences=False))

    # Fully-connected layer
    model.add(Dense(inputMLP, activation=activationMLP))

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    # Print the summary of the model
    #model.summary

    # Return the model
    return model




#-------------------------------------------------------------------------------------------------------
# normConv2Lstm: define a batch normalization + 2 convolutional/max-pooling + LSTM DNN
#-------------------------------------------------------------------------------------------------------
def normConv2Lstm(
    inputShape,
    nkerns,
    filterSizes,
    poolSizes,
    activationConv,
    outputLSTM,
    inputMLP,
    activationMLP,
    nbClasses,
    withSoftmax=True):

    outputSizeLastConv = ((inputShape[0]-filterSizes[0][0]+1)/poolSizes[0][0]-filterSizes[1][0]+1)/poolSizes[1][0]

    model = Sequential()

    # Batch normalization layer
    model.add(BatchNormalization(input_shape=inputShape))

    # 1st convolutional + max-pooling layer
    model.add(Conv2D(nkerns[0], kernel_size=filterSizes[0], activation=activationConv))
    model.add(MaxPooling2D(pool_size=poolSizes[0]))

    # 2nd convolutional + max-pooling layer
    model.add(Conv2D(nkerns[1], kernel_size=filterSizes[1], activation=activationConv))
    model.add(MaxPooling2D(pool_size=poolSizes[1]))
    print(model.layers[-1].output_shape)

    # LSTM layer with a many-to-one implementation
    # Note: size of the output = (outputSizeLastConv, outputLSTM)
    model.add(Reshape((outputSizeLastConv,inputShape[1]*nkerns[1])))
    model.add(LSTM(outputLSTM))
  
    # Fully-connected layer
    model.add(Dense(inputMLP, activation=activationMLP))

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    # Return the model
    return model



#-------------------------------------------------------------------------------------------------------
# normConv3Lstm: define a batch normalization + 3 convolutional/max-pooling + LSTM DNN
#-------------------------------------------------------------------------------------------------------
def normConv3Lstm(
    inputShape,
    nkerns,
    filterSizes,
    poolSizes,
    activationConv,
    outputLSTM,
    inputMLP,
    activationMLP,
    nbClasses,
    withSoftmax=True):

    outputSizeLastConv = (((inputShape[0]-filterSizes[0][0]+1)/poolSizes[0][0]-filterSizes[1][0]+1)/poolSizes[1][0]) - filterSizes[2][0] + 1

    model = Sequential()

    # Batch normalization layer
    model.add(BatchNormalization(input_shape=inputShape))

    # 1st convolutional + max-pooling layer
    model.add(Conv2D(nkerns[0], kernel_size=filterSizes[0], activation=activationConv))
    model.add(MaxPooling2D(pool_size=poolSizes[0]))

    # 2nd convolutional + max-pooling layer
    model.add(Conv2D(nkerns[1], kernel_size=filterSizes[1], activation=activationConv))
    model.add(MaxPooling2D(pool_size=poolSizes[1]))

    # 3rd convolutional + RELU
    model.add(Conv2D(nkerns[2], kernel_size=filterSizes[2], activation=activationConv))


    # Reshaping layer
    model.add(Reshape((outputSizeLastConv,inputShape[1]*nkerns[2])))
    #print(model.layers[-1].output_shape)

    # LSTM layer with a many-to-one implementation
    # Note: size of the output = (outputSizeLastConv, outputLSTM)
    model.add(LSTM(outputLSTM))
  
    # Fully-connected layer
    #model.add(Flatten())
    model.add(Dense(inputMLP, activation=activationMLP))
    #model.add(Dropout(0.5))

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    # Return the model
    return model


#-------------------------------------------------------------------------------------------------------
# normConv3Lstm2: define a batch normalization + 3 convolutional/max-pooling + 2 LSTM layers DNN
#-------------------------------------------------------------------------------------------------------
def normConv3Lstm2(
    inputShape,
    nkerns,
    filterSizes,
    poolSizes,
    activationConv,
    outputLSTM1,
    outputLSTM2,
    inputMLP,
    activationMLP,
    nbClasses,
    withSoftmax=True):

    outputSizeLastConv = (((inputShape[0]-filterSizes[0][0]+1)/poolSizes[0][0]-filterSizes[1][0]+1)/poolSizes[1][0]) - filterSizes[2][0] + 1

    model = Sequential()

    # Batch normalization layer
    model.add(BatchNormalization(input_shape=inputShape))

    # 1st convolutional + max-pooling layer
    model.add(Conv2D(nkerns[0], kernel_size=filterSizes[0], activation=activationConv))
    model.add(MaxPooling2D(pool_size=poolSizes[0]))

    # 2nd convolutional + max-pooling layer
    model.add(Conv2D(nkerns[1], kernel_size=filterSizes[1], activation=activationConv))
    model.add(MaxPooling2D(pool_size=poolSizes[1]))

    # 3rd convolutional + RELU
    model.add(Conv2D(nkerns[2], kernel_size=filterSizes[2], activation=activationConv))


    # Reshaping layer
    model.add(Reshape((outputSizeLastConv,inputShape[1]*nkerns[2])))

    # LSTM layer with a many-to-one implementation
    model.add(LSTM(outputLSTM1,return_sequences=True))
    model.add(LSTM(outputLSTM2))
  
    # Fully-connected layer
    model.add(Dense(inputMLP, activation=activationMLP))

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    # Return the model
    return model

#-------------------------------------------------------------------------------------------------------
# bidirectionalLSTM: define a batch normalization + bidirectional LSTM DNN
#-------------------------------------------------------------------------------------------------------
def bidirectionalLSTM(
    inputShape,
    outputLSTM,
    inputMLP,
    activationMLP,
    nbClasses,
    withSoftmax=True):

    ### Model definition
    model = Sequential()

    # Batch normalization layer
    model.add(BatchNormalization(input_shape=inputShape))
    print(model.layers[-1].output_shape)

    # Bi-directional LSTM layer
    model.add(Reshape((inputShape[0],inputShape[1])))
    model.add(Bidirectional(LSTM(outputLSTM,return_sequences=True),input_shape=(inputShape[0],inputShape[1])))
    
    # Dense layer
    model.add(Flatten())
    model.add(Dense(inputMLP,activation=activationMLP))
    
    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    # Return the model
    return model



#-------------------------------------------------------------------------------------------------------
# bidirectionalLSTM2: define a batch normalization + 2 bidirectional LSTM DNN
#-------------------------------------------------------------------------------------------------------
def bidirectionalLSTM2(
    inputShape,
    outputLSTM,
    inputMLP,
    activationMLP,
    nbClasses,
    withSoftmax=True):
    
    ### Model definition
    model = Sequential()

    # Batch normalization layer
    model.add(BatchNormalization(input_shape=inputShape))

    # Bi-directional LSTM
    model.add(Reshape((inputShape[0],inputShape[1])))
    model.add(Bidirectional(LSTM(outputLSTM,return_sequences=True),input_shape=(inputShape[0],inputShape[1])))
    model.add(Bidirectional(LSTM(outputLSTM)))

    # Dense layer
    model.add(Dense(inputMLP,activation=activationMLP))
   
    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    # Return the model
    return model



#-------------------------------------------------------------------------------------------------------
# convAutoencoder: define a convolutional autoencoder model with conv+pooling layers
#-------------------------------------------------------------------------------------------------------
def convAutoencoder(
    inputShape,
    nkerns,
    filterSizes,
    poolSizes,
    activationConv,
    activationMLP,
    decoder=True):

    #outputSizeLastConv = (inputShape[0]-filterSizes[0][0]+1)/poolSizes[0][0]
    #inputMLP = nkerns[0]*outputSizeLastConv*inputShape[1]

    # NOTE: if padding = 'same', the size of the feature maps doesn't change after being convoluted
    outputSizeLastConv = inputShape[0]/poolSizes[0][0]
    inputMLP = nkerns[0]*outputSizeLastConv*inputShape[1]

    model = Sequential()

    # First convolutional layer
    # Note: default padding = zero padding?
    model.add(Conv2D(nkerns[0], kernel_size=filterSizes[0], activation=activationConv, padding='same', input_shape=inputShape))

    # Max-pooling layer
    model.add(MaxPooling2D(pool_size=poolSizes[0], padding='same'))


    # Dense layer
    model.add(Flatten())
    model.add(Dense(inputMLP, activation=activationMLP))


    if decoder:

        # Reshaping layer
        model.add(Reshape((outputSizeLastConv,inputShape[1],nkerns[0])))

        # Up-sampling layer
        model.add(UpSampling2D(size=poolSizes[0]))

        # Second deconvolutional layer
        model.add(Conv2D(nkerns[0], kernel_size=filterSizes[0], activation=activationConv, padding='same'))

        # Output layer
        model.add(Conv2D(1, kernel_size=filterSizes[0], activation='linear', padding='same')) 

    return model


#-------------------------------------------------------------------------------------------------------
# autoencoder1: define a autoencoder with 1 hidden layer
#-------------------------------------------------------------------------------------------------------
def autoencoder1(
    inputShape,
    inputMLP,
    activationMLP,
    decoder=True):

    model = Sequential()

    # Dense layer of the encoder
    model.add(Dense(inputMLP,activation=activationMLP,input_shape=inputShape))
    print('Encoder dense shape: ------------------------------')
    print(model.layers[-1].output_shape)

    if decoder:

        # Dense layer of the decoder
        model.add(Dense(inputMLP,activation=activationMLP))
        print('Decoder dense shape: ------------------------------')
        print(model.layers[-1].output_shape)

        # Output layer
        model.add(Dense(inputShape[0],activation='linear')) 
        print('Output shape: ------------------------------')
        print(model.layers[-1].output_shape)      

    return model


#-------------------------------------------------------------------------------------------------------
# autoencoder2: define a autoencoder with 2 hidden layers
#-------------------------------------------------------------------------------------------------------
def autoencoder2(
    inputShape,
    inputMLP1,
    inputMLP2,
    activationMLP,
    decoder=True):

    model = Sequential()

    # Dense layer 1 of the encoder
    model.add(Dense(inputMLP1,input_shape=inputShape,activation=activationMLP))

    # Dense layer 2 of the encoder
    model.add(Dense(inputMLP2,activation=activationMLP))

    if decoder:

        # Dense layer 1 of the decoder
        model.add(Dense(inputMLP2,activation=activationMLP))
        
        # Dense layer 2 of the decoder
        model.add(Dense(inputMLP1,activation=activationMLP))
        
        # Output layer
        model.add(Dense(inputShape[0],activation='linear')) 

    return model



#-------------------------------------------------------------------------------------------------------
# autoencoder3: define a autoencoder with 3 hidden layers
#-------------------------------------------------------------------------------------------------------
def autoencoder3(
    inputShape,
    inputMLP1,
    inputMLP2,
    inputMLP3,
    activationMLP,
    decoder=True):

    model = Sequential()

    # Dense layer 1 of the encoder
    model.add(Dense(inputMLP1,activation=activationMLP,input_shape=inputShape))

    # Dense layer 2 of the encoder
    model.add(Dense(inputMLP2,activation=activationMLP))

    # Dense layer 3 of the encoder
    model.add(Dense(inputMLP3,activation=activationMLP))

    if decoder:

        # Dense layer 1 of the decoder
        model.add(Dense(inputMLP3,activation=activationMLP))

        # Dense layer 1 of the decoder
        model.add(Dense(inputMLP2,activation=activationMLP))

        # Dense layer 2 of the decoder
        model.add(Dense(inputMLP1,activation=activationMLP))

        # Output layer
        model.add(Dense(inputShape[0],activation='linear')) 

    return model