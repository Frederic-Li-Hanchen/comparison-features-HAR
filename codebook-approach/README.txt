#################################################################################################################
         "Codebook-based feature extraction code" used in the paper
         Comparison of Feature Learning Methods for Human Activity Recognition using Wearable Sensors
         
                        F. Li, K. Shirahama, M. A. Nisar, L. Koeping, M. Grzegorzek
                              Research Group for Pattern Recognition
                                   University of Siegen, Germany
#################################################################################################################

The codebook-based feature extraction code has been written in C++. To load and output python's numpy format
files, the code uses "numpy.hpp" which is a header-only library[1] to intermedate between C++ and numpy files.

For any questions related to this code, please contact Kimiaki Shirahama at: kimiaki.shirahama@uni-siegen.de

###########################
# How to compile the code #
###########################
make
As a result, the executable file named "main" will be generated.

#######################
# How to run the code #
#######################

The following rough description is displayed by running the code with no option.

Usage: ./main -d <Directory of *_data.npy files> -w <Window size> -l <Sliding stride size> -c <Directory of codebook files> -n <# of codewords per codebook> -o <Directory of output feature files>
 *** The followings are optional ***
 -s <Smoothing parameter (default -1: A value <= 0 means hard assignment, otherwise soft assignment)>
 -st <ID of the file (*_data.npy) from which feature encoding starts>
     (default -1 meaning that feature encoding will start from the first (0th) file)
 -ed <ID of the file (*_data.npy) at which feature econding ends>
     (default -1 meaning that feature encoding will end at the last file>

More specifically, the code supports two cases. The first case is called "multiple sensor case" and targets multiple sensor channels like the OPPORTUNITY dataset case. Given an example (i.e. data frame), the feature for each sensor channel is extracted using the codebook, which has been constructed beforehand using data on this sensor channel. Then, early fusion is performed to fuse such features for all sensor channels into a signle high-dimensional feature. The second case is called "single sensor case" and targets to extract a feature for a signle sensor channel like the UniMiB-SHAR dataest case. Here, the feature of an example is extracted using one codebook, which has been constructed beforehand on the sensor channel. 

# Example commands for the multiple sensor case #
Hard assignment: ./main -d <Directory of input *_data.npy files for the OPPORTUNITY dataset> -w 24 -l 1 -c ./codebooks/OPPORTUNITY/w24_l1_wn128/ -n 128 -o <Output directory>
Soft assignment: ./main -d <Directory of input *_data.npy files for the OPPORTUNITY dataset> -w 24 -l 1 -c ./codebooks/OPPORTUNITY/w24_l1_wn128/ -s 256 -n 128 -o <Output directory>

As a result, .npy files with the same names to the inputs are stored in <Output directory>, for instance, features for S1-ADL1_data.npy are stored in <Output directory>/S1-ADL1_data.npy. For the hard assigment case, such files with big sizes can be transformed into "sparse" ones using "converToSparse.py" under the directory "OPPORTUNITY/features/CBh/". For the soft assigment case, features files for training and those for testing can be obtained using "OPPORTUNITY/features/CBs/combineFiles.py". These python codes are very simple, so you should need no further explanation.

Finally, the code assumes that codebooks for different sensor channels are stored under <Directory of codebook files> using the file name format, "codebook_s0.txt", "codebook_s1.txt", "codebook_s2.txt", and so on.

# Example commands for the single sensor case #
Hard assignment: ./main -d <Directory of input *_data.npy files for the UniMiB-SHAR dataset> -w 16 -l 1 -c ./codebooks/UniMiB-SHAR/w16_l1_wn2048/ -n 2048 -o <Output directory>
Soft assignment: ./main -d <Directory of input *_data.npy files for the UniMiB-SHAR dataset> -w 16 -l 1 -c ./codebooks/UniMiB-SHAR/w16_l1_wn2048/ -s 4 -n 2048 -o <Output directory>

Like the multi sensor case, .npy files with the same names to the inputs are stored in <Output directory>. For the experiments in our paper, we didn't need to combine feature files, so we don't include any python code under "UniMiB-SHAR/features/CBh" or "UniMiB-SHAR/features/CBs". Combining features can be easily done by directory using or slightly modifiying the pythonn codes for the OPPORTUNITY dataset.

Finally, in this single sensor case, there is only one codebook file under <Directory of codebook files>. The code switches between the single-sensor and multiple-sensor cases based on the number of codebook files under <Directory of codebook files>.

[1] http://mglab.blogspot.de/2014/03/numpyhpp-simple-header-only-library-for_9560.html
