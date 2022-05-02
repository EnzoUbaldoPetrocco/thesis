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


####################################################################
###################### PLOT IMAGE ##################################
print('PLOT IMAGE')
plt.figure()
plt.imshow(np.reshape(FX[30], (200,200)))

####################################################################
################### NORMALIZE DATA #################################
print('NORMALIZE DATA')
scalerX = preprocessing.MinMaxScaler()
FX = scalerX.fit_transform(FX)
FXT = scalerX.transform(FXT)

#####################################################################
################### MODEL SELECTION (HYPERPARAMETER TUNING)##########
print('MODEL SELECTION AND TUNING')
Fgrid = {'C':        np.logspace(-4,3,5),
        'kernel':   ['rbf'],
        'gamma':    np.logspace(-4,3,5)}
FMS = GridSearchCV(estimator = SVC(),
                  param_grid = Fgrid,
                  scoring = 'balanced_accuracy',
                  cv = 10,
                  verbose = 0)
FH = FMS.fit(FX,FY)
print('CLASSIFICATION')
FM = SVC(C = FH.best_params_['C'],
        kernel = FH.best_params_['kernel'],
        gamma = FH.best_params_['gamma'])
FM.fit(FX,FY)

####################################################
################## TESTING #########################

print('RPREDICTING FRENCH TEST SET')
FYF = FM.predict(FXT)
confusion_matrix(FYT,FYF)
print('PREDICTING CHINESE TEST SET')
CFYF = FM.predict(CXT)
confusion_matrix(CYT,CFYF)


print('arrivato')