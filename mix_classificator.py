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
'''ix = range(0,247)
iy = 247
L = pd.read_csv('drive/MyDrive/Didattica/VariousDatasets/CRIME/L.csv')
T = pd.read_csv('drive/MyDrive/Didattica/VariousDatasets/CRIME/T.csv')
L = L.to_numpy()
T = T.to_numpy()

X = L[:,ix]
Y = L[:,iy]
XT = T[:,ix]
YT = T[:,iy]'''
'''
## CDS stands for chinese dataset, FDS stands for french dataset
CDS = pd.read_csv('./chinese/chinese.csv').to_numpy()
FDS = pd.read_csv('./french/french.csv').to_numpy()

## I need to exctract casually 70% of samples for training set and 30% of dataset for test set
## This must be done for both datasets
random.seed(11)

##Chinese part
test_CDS = []
for i in CDS:
    test_CDS.append(i)
percentage_chinese = len(CDS)
percentage_chinese_training = int(percentage_chinese * 0.7)
percentage_chinese_test = percentage_chinese - percentage_chinese_training
print(test_CDS[2][0])
print(type(test_CDS[2][0]))
#decode_string_matrix.MatrixDecoder.decoder(test_CDS[2][0])
#print(decode_string_matrix.MatrixDecoder.decoder(test_CDS[2][0]))
#training set
CX = []
CXT = []
CY = []
CYT = []
count = 0
for i in range(0,percentage_chinese_training):
    index = random.randint(0,percentage_chinese-count)
    #TCDS.append(test_CDS[index])
    CX.append(test_CDS[index][0])
    CX.append(test_CDS[index][1])
    test_CDS.pop(index)
    count = count +1
for i in range(0,len(test_CDS)):
    CY.append(test_CDS[i][0].to_numpy())
    CYT.append(test_CDS[i][1])

## French part
test_FDS = []
for i in FDS:
    test_FDS.append(i)
percentage_french = len(FDS)
percentage_french_training = int(percentage_french * 0.7)
percentage_french_test = percentage_french - percentage_french_training
#training set

FX = []
FY = []
FXT = []
FYT = []

count = 0
for i in range(0,percentage_french_training):
    index = random.randint(0,percentage_french-count)
    FX.append(test_FDS[index][0].to_numpy())
    FX.append(test_FDS[index][1])
    test_FDS.pop(index)
    count = count +1

for i in range(0,len(test_CDS)):
    FY.append(test_CDS[i][0].to_numpy())
    FYT.append(test_CDS[i][1])

'''

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
################### NORMALIZE DATA #################################

scalerX = preprocessing.MinMaxScaler()
CX = scalerX.fit_transform(CX)
CXT = scalerX.transform(CXT)
FX = scalerX.fit_transform(FX)
FXT = scalerX.transform(FXT)

#####################################################################
################### MODEL SELECTION (HYPERPARAMETER TUNING)##########
Cgrid = {'C':        np.logspace(-4,3,5),
        'kernel':   ['rbf'],
        'gamma':    np.logspace(-4,3,5)}
CMS = GridSearchCV(estimator = SVC(),
                  param_grid = Cgrid,
                  scoring = 'balanced_accuracy',
                  cv = 10,
                  verbose = 0)
CH = CMS.fit(CX,CY)


Fgrid = {'C':        np.logspace(-4,3,5),
        'kernel':   ['rbf'],
        'gamma':    np.logspace(-4,3,5)}
FMS = GridSearchCV(estimator = SVC(),
                  param_grid = Fgrid,
                  scoring = 'balanced_accuracy',
                  cv = 10,
                  verbose = 0)
FH = FMS.fit(FX,FY)


CM = SVC(C = CH.best_params_['C'],
        kernel = CH.best_params_['kernel'],
        gamma = CH.best_params_['gamma'])
CM.fit(CX,CY)

FM = SVC(C = FH.best_params_['C'],
        kernel = FH.best_params_['kernel'],
        gamma = FH.best_params_['gamma'])
FM.fit(FX,FY)

####################################################
################## TESTING #########################

CYF = CM.predict(CXT)
confusion_matrix(CYT,CYF)

FYF = FM.predict(FXT)
confusion_matrix(FYT,FYF)

print('arrivato')