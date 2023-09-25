#################################################################################################################
         "Hand-crafted feature extraction code" used in the paper
         Comparison of Feature Learning Methods for Human Activity Recognition using Wearable Sensors
         
                        F. Li, K. Shirahama, M. A. Nisar, L. Koeping, M. Grzegorzek
                              Research Group for Pattern Recognition
                                   University of Siegen, Germany
#################################################################################################################



The hand-crafted feature extraction code has been written in Python. We use four Python files for this purpose. Each file contains variables to set the paths of input and output files. 

For any questions related to this code, please contact Muhammad Adeel Nisar at: Adeel.Nisar@uni-siegen.de


#######################
# How to run the code #
#######################

Modify the hyper-parameters at the beginning of each file.

Then execute the programs in the following order:

1. generateAndSaveFeatures.py:  uses sensor data to generate 18 hand-crafted features and saves them on the specified path.
2. makeFeatureMatrices.py:      organizes training and testing data in the form of matrices.
3. normalizeFeatureMatrices.py: normalizes the data.
4. svmClassify.py:              trains model and generates results. 
