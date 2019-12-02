import argparse
import numpy as np
import os
import fnmatch

from sklearn.svm import NuSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

"""
Runs 5-fold CV on the schizophrenia dataset and calls the 
run_exp_noLR_idvclass.py file modified to work with the schizophrenia data
"""

if not os.path.exists('input/CV'):
    os.mkdir('input/CV')
    print("Directory " , 'input/CV' ,  " Created ")

k = 2

X = np.load('input/tr_data.npy')
y = np.zeros(shape=(X.shape[0],1))

for i in range(X.shape[0]):
    if i < X.shape[0] / 2:
        np.append(y, 0)
    else:
        np.append(y, 1)

skf = StratifiedKFold(n_splits=k)

np.save('input/CV/tr+tst_data.npy', X)

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y[train_index], y[test_index]

    np.save('input/CV/tr_data.npy', X_train)
    np.save('input/CV/tst_data.npy', X_test)

    # Execute run_exp_noLR_idvclass.py using current train and test data
    os.system("python " + "run_exp_noLR_idvclas.py " + "input/CV/ " + "38 137 idvclas_svm srm --loo 2  1 38 38")


    
