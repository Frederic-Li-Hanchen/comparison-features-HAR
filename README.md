# Comparison of feature extraction approaches for wearable-based human activity recognition

This repository hosts the code that was used for the publication:

**[*]** Frédéric Li, Kimiaki Shirahama, Muhammad Adeel Nisar, Lukas Köping, Marcin Grzegorzek, **Comparison of Feature Learning Methods for Human Activity Recognition using Wearable Sensors**, _Sensors_ (MDPI), Vol. 18, Issue 2, 2018, https://doi.org/10.3390/s18020679


The repository contains Python and C++ implementations of the various feature extraction techniques used in the study of **[*]**. It is structured into the following subfolders:
- **./codebook-approach**: C++ implementation of the codebook approach (written by Prof. Dr. Kimiaki Shirahama: https://ccilab.doshisha.ac.jp/shirahama/index_en.html)
- **./data-preprocessing**: various Python scripts to pre-process the datasets used in the study
- **./deep-learning**: Python implementation of the deep feature learning approaches used in the study
- **./dnn-cb-fusion**: Python implementation of the fusion between deep and codebook features
- **./hand-crafted**: Python implementation of the hand-crafted features used in the study (written by Dr.-Ing. Adeel Nisar: http://pu.edu.pk/faculty/description/975/Dr-Muhammad-Adeel-Nisar.html)


Each subfolder contains a README file describing the contents of each script it contains. The data used in the study can be found at the following repository: https://ccilab.doshisha.ac.jp/shirahama/research/feature_learning/
