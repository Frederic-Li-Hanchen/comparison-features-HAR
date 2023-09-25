#################################################################################################################
        Comparison of Feature Learning Methods for Human Activity Recognition using Wearable Sensors [*]
         
                        F. Li, K. Shirahama, M. A. Nisar, L. Koeping, M. Grzegorzek
                              Research Group for Pattern Recognition
                                   University of Siegen, Germany
#################################################################################################################

This folder contains all the codes required to train deep-learning models, compute features and train a SVM classifier on the OPPORTUNITY and UniMiB-SHAR datasets.
All Python codes provided here are using the Keras [1] deep learning library with a Tensorflow [2] back-end.
For any questions realated to this code, please contact Frederic Li at: frederic.li.hanchen@gmail.com


##################
# IMPORTANT NOTE # 
##################

There are currently two versions of the code, one for each dataset.
Both versions notably feature a lot of code duplication.
A "cleaner" version of the code involving some code refactoring will be uploaded in the future.


###############
# OPPORTUNITY #
###############

This subfolder contains two utilitary files:
    - utilitary.py: various functions and attributes used by the other scripts.
    - dnnModels.py: Keras definition of the different deep-learning models.
    
and three scripts:
    - trainDnn.py: train a deep-learning model. The trained model is saved in a HDF5 file (.h5).
    - extractDnnFeatures.py: compute deep-learning features on a dataset using a trained model. The features are saved as .npy files.
    - svmClassify.py: train and evaluate a soft-margin linear SVM classifier using the pre-computed features.
    
Usage:
    1) Edit the hyper-parameters at the beginning of the trainDnn.py script.
    2) Execute in order the three scripts using:
          -> python trainDnn.py
          -> python extractDnnFeatures.py
          -> python svmClassify.py
  
          
###############
# UniMiB-SHAR #
###############

This subfolder contains three utilitary files:
    - utilitary.py: various functions and attributes used by the other scripts.
    - dnnModels.py: Keras definition of the different deep-learning models.
    - synthetizeCvResults.py: displays the results of the Leave-One-Subject-Out Cross Validation and saves them in a .csv file.
    
three Python scripts:
    - trainDnn.py: train a deep-learning model. The trained model is saved in a HDF5 file (.h5).
    - extractDnnFeatures.py: compute deep-learning features on a dataset using a trained model. The features are saved as .npy files.
    - svmClassify.py: train and evaluate a soft-margin linear SVM classifier using the pre-computed features.
    
and one bash script:
    - losocv_main.sh: train and evaluate a soft-margin linear SVM using deep-learning features for all 30 subjects of the dataset.
    
Usage for the 20-train/10-test data case:
    1) Edit the hyper-parameters at the beginning of each of the three scripts (trainDnn.py, extractDnnFeatures.py, svmClassify.py)
    2) Execute in order the three scripts using:
        -> python trainDnn.py
        -> python extractDnnFeatures.py
        -> python svmClassify.py
        
Usage for the one-subject test/other subjects train case:        
    1) Edit the hyper-parameters at the beginning of each of the three scripts (trainDnn.py, extractDnnFeatures.py, svmClassify.py)
    2) Execute in order the three scripts precising the testing subject index using:
        -> python trainDnn.py n
        -> python extractDnnFeatures.py n
        -> python svmClassify.py n
    with n between 1 and 30, index of the testing subject.
    
Usage for the Leave-One-Subject-Out Cross Validation case:
    1) Edit the hyper-parameters at the beginning of each of the three scripts (trainDnn.py, extractDnnFeatures.py, svmClassify.py)
    2) Edit the result path in the synthetizeCvResults.py script
    3) Execute the bash script with -> ./losocv_main.sh
    4) Execute the synthetizeCvResults.py script with -> python synthetizeCvResults.py
    
    
NOTE (in the case of the Leave-One-Subject-Out Cross Validation): 
The current version of the code overwrites models and features computed on previous folds.
The code can be modified to save all of them by changing the names of the saved models and features.


##############
# References #
##############

[1] F. Chollet et al., Keras, https://github.com/fchollet/keras

[2] I. Goodfellow et al. TensorFlow: Large-scale machine learning on heterogeneous systems. 
    http: //tensorflow.org/, 2015. Software available from tensorflow.org.
