#################################################################################################################
        Comparison of Feature Learning Methods for Human Activity Recognition using Wearable Sensors [*]
         
                        F. Li, K. Shirahama, M. A. Nisar, L. Koeping, M. Grzegorzek
                              Research Group for Pattern Recognition
                                   University of Siegen, Germany
#################################################################################################################


This folder contains various scripts used to perform pre-processing and segmentation operations on both OPPORTUNITY and UniMiB-SHAR datasets.
For any questions about them, please contact Frederic Li at: frederic.li@uni-siegen.de

###############
# OPPORTUNITY #
###############

cleanOpportunityData.py: remove corrupted sensor channels and values from the original .dat data files. The "cleaned" data files are saved as .txt.

computeTimeWindows.py: extract time windows and labels from the cleaned data files, and save them as Python files (.npy).


###############
# UniMiB-SHAR #
###############

build20Train10Test.py: build the custom training and testing sets using respectively the 20 first and 10 last subjects. The data is saved as .npy files.

separateDataPerSubject.py: split the orginal MATLAB data and labels per subject. For each subject, two .npy files containing data frames and labels are created.


#########
# Usage #
#########

1) Modify the hyper parameters (e.g. data paths, window size ...) at the beginning of the script to use.

2) Execute the Python script with > python script_name.py