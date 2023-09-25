#################################################################################################################
        Comparison of Feature Learning Methods for Human Activity Recognition using Wearable Sensors [*]
         
                        F. Li, K. Shirahama, M. A. Nisar, L. Koeping, M. Grzegorzek
                              Research Group for Pattern Recognition
                                   University of Siegen, Germany
#################################################################################################################

This folder contains the code to test the codebook and deep-learning fusion approach presented in [*].

Usage: 
    1) Change the hyper-parameters at the beginning of dnnCbSvmFusion.py
       NOTE: the paths to the SVM scores obtained by CB and Hybrid have to be indicated.
             It is possible to save those SVM scores by uncommenting the relevant lines in the svmClassify.py scripts.
    2) Execute the script with -> python dnnCbSvmFusion.py
    
For more questions about this code, please contact Frederic Li at: frederic.li.hanchen@gmail.com
