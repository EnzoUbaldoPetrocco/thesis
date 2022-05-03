#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import manipulating_images
from math import floor
from sklearn.metrics import confusion_matrix
############################################################
############### READ DATA ##################################
itd = manipulating_images.ImagesToData()

CX = itd.chinese[0:floor(len(itd.chinese)*0.7)]
CXT = itd.chinese[floor(len(itd.chinese)*0.7):len(itd.chinese)-1]
CY = itd.chinese_categories[0:floor(len(itd.chinese)*0.7)]
CYT = itd.chinese_categories[floor(len(itd.chinese)*0.7):len(itd.chinese)-1]

FX = itd.french[0:floor(len(itd.french)*0.7)]
FXT = itd.french[floor(len(itd.french)*0.7):len(itd.french)-1]
FY = itd.french_categories[0:floor(len(itd.french)*0.7)]
FYT = itd.french_categories[floor(len(itd.french)*0.7):len(itd.french)-1]

MX = np.concatenate((CX, FX), axis=0)
MXT = np.concatenate((CXT, FXT), axis = 0)
MY = np.concatenate((CY, FY), axis = 0)
MYT = np.concatenate((CYT, FYT), axis=0)

####################################################################
###################### PLOT IMAGE ##################################
print('PLOT IMAGE')
plt.figure()
plt.imshow(np.reshape(CX[30], (250,250)))
plt.show()
plt.figure()
plt.imshow(np.reshape(FX[30], (250,250)))
plt.show()
####################################################################
################### NORMALIZE DATA #################################
print('NORMALIZE DATA')
scalerX = preprocessing.MinMaxScaler()
MX = scalerX.fit_transform(MX)
MXT = scalerX.transform(MXT)

#####################################################################
################### MODEL SELECTION (HYPERPARAMETER TUNING)##########
print('MODEL SELECTION AND TUNING')
Mgrid = {'C':        np.logspace(-4,3,5),
        'kernel':   ['rbf'],
        'gamma':    np.logspace(-4,3,5)}
MMS = GridSearchCV(estimator = SVC(),
                  param_grid = Mgrid,
                  scoring = 'balanced_accuracy',
                  cv = 10,
                  verbose = 0)
MH = MMS.fit(CX,CY)

print('CLASSIFICATION')
MM = SVC(C = MH.best_params_['C'],
        kernel = MH.best_params_['kernel'],
        gamma = MH.best_params_['gamma'])
MM.fit(CX,CY)


####################################################
################## TESTING #########################

print('RPREDICTING FRENCH TEST SET')
MFYF = MM.predict(FXT)
print(confusion_matrix(FYT,MFYF))
print('PREDICTING CHINESE TEST SET')
MCYF = MM.predict(CXT)
print(confusion_matrix(CYT,MCYF))
print('PREDICTING MIX TEST SET')
MCYF = MM.predict(MXT)
print(confusion_matrix(MYT,MCYF))


print('arrivato')