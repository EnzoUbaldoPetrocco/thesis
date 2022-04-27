#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import random
import time

## READ DATA
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

## CDS stands for chinese dataset, FDS stands for french dataset
CDS = pd.read_csv('./chinese/chinese.csv').to_numpy()
FDS = pd.read_csv('./french/french.csv').to_numpy()

## I need to exctract casually 70% of samples for training set and 30% of dataset for test set
## This must be done for both datasets
random.seed(11)
print(CDS)
#test set
test_CDS = CDS
percentage_chinese = (len(CDS)-1)/2
percentage_chinese_training = int(percentage_chinese * 0.7)
percentage_chinese_test = percentage_chinese - percentage_chinese_training
print(percentage_chinese)
#training set
TCDS = []

for i in range(0,percentage_chinese_training):
    index = random.randint(0,percentage_chinese)
    TCDS.append(test_CDS[index*2])
    np.delete(TCDS,index*2 )
    #del test_CDS[index*2]
for i in range(0,(len(test_CDS)/2)):
    #del test_CDS[i*2]
    np.delete(TCDS,i*2 )


