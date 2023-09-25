
from __future__ import print_function

import numpy as np
import sys
import glob
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack

def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def convertToSparseMatrix(featDir, featFilesDense, combSparseFeatFilename):

    featFilesSparse = []
    for fn in featFilesDense:
        print(">> Loading the denes feature matrix {0}".format(fn))
        featDense = np.load(featDir + fn)
        featSparse = csr_matrix(featDense)
        fnSparse = fn.replace(".npy", ".npz")
        print(">> Saving the sparse feature matrix {0}".format(fnSparse))
        save_npz(featDir + fnSparse, featSparse)
        featFilesSparse.append(fnSparse)
        # set_trace()

    nbFeatFiles = len(featFilesSparse)
    featAllSparse = load_npz(featDir + featFilesSparse[0])
    for idx in range(1, nbFeatFiles):
        featSparse = load_npz(featDir + featFilesSparse[idx])
        print(
            ">> Concatenating the {0}th sparse matrix {1}".
            format(idx, featFilesSparse[idx])
            )
        featAllSparse = vstack([featAllSparse, featSparse])
        print("---> {0}".format(featAllSparse.shape))
    print(">> Save the combined sparse matrix into {0}".format(combSparseFeatFilename))
    save_npz(featDir + combSparseFeatFilename, featAllSparse)


def usage():
    print("Usage: python convertToSparse.py <Directory of feature data>")


if __name__ == '__main__':

    params = sys.argv
    if len(params) != 2:
        usage()
        sys.exit()

    print('----------------------------------------------------------')
    print('Convert dense feature matrixes under {0} into sparse ones'.format(params[1]))
    print('----------------------------------------------------------')

    trainingFeatFiles = [
        "S1-ADL1_data.npy", "S1-ADL2_data.npy", "S1-ADL3_data.npy", "S1-Drill_data.npy",
        "S2-ADL1_data.npy", "S2-ADL2_data.npy", "S2-ADL3_data.npy", "S2-Drill_data.npy",
        "S3-ADL1_data.npy", "S3-ADL2_data.npy", "S3-ADL3_data.npy", "S3-Drill_data.npy",
        "S4-ADL1_data.npy", "S4-ADL2_data.npy", "S4-ADL3_data.npy", "S4-Drill_data.npy"
    ]
    convertToSparseMatrix(params[1], trainingFeatFiles, "Features_training")

    testingFeatFiles = [
        "S1-ADL4_data.npy", "S1-ADL5_data.npy",
        "S2-ADL4_data.npy", "S2-ADL5_data.npy",
        "S3-ADL4_data.npy", "S3-ADL5_data.npy",
        "S4-ADL4_data.npy", "S4-ADL5_data.npy",
    ]
    convertToSparseMatrix(params[1], testingFeatFiles, "Features_testing")
