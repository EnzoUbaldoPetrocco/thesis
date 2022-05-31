#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import manipulating_images_better
from math import floor
from sklearn.metrics import confusion_matrix


# Confusion matrix lists
Ccm_list = []
Fcm_list = []
Mcm_list = []

for i in range(30):
        print('CICLE: ' + str(i))

        ############################################################
        ############### READ DATA ##################################
        itd = manipulating_images_better.ImagesToData()
        itd.bf_ml()

        CX = itd.chinese[0:floor(len(itd.chinese)*0.7)]
        CXT = itd.chinese[floor(len(itd.chinese)*0.7):len(itd.chinese)-1]
        CY = itd.chinese_categories[0:floor(len(itd.chinese)*0.7)]
        CYT = itd.chinese_categories[floor(len(itd.chinese)*0.7):len(itd.chinese)-1]

        FX = itd.french[0:floor(len(itd.french)*0.7)]
        FXT = itd.french[floor(len(itd.french)*0.7):len(itd.french)-1]
        FY = itd.french_categories[0:floor(len(itd.french)*0.7)]
        FYT = itd.french_categories[floor(len(itd.french)*0.7):len(itd.french)-1]

        MX = itd.mixed[0:floor(len(itd.mixed)*0.7)]
        MXT = itd.mixed[floor(len(itd.mixed)*0.7):len(itd.mixed)-1]
        MY = itd.mixed_categories[0:floor(len(itd.mixed)*0.7)]
        MYT = itd.mixed_categories[floor(len(itd.mixed)*0.7):len(itd.mixed)-1]
        

        ####################################################################
        ################### NORMALIZE DATA #################################
        #print('NORMALIZE DATA')
        #scalerX = preprocessing.MinMaxScaler()
        #FX = scalerX.fit_transform(FX)
        #FXT = scalerX.transform(FXT)

        #####################################################################
        ################### MODEL SELECTION (HYPERPARAMETER TUNING)##########
        print('MODEL SELECTION AND TUNING')
        Fgrid = {'C':        np.logspace(-1,5,30),
                'kernel':   ['rbf'],
                'gamma':    np.logspace(-5,1,30)}
        FMS = GridSearchCV(estimator = SVC(),
                        param_grid = Fgrid,
                        scoring = 'balanced_accuracy',
                        cv = 10,
                        verbose = 0)
        FH = FMS.fit(FX,FY)
        print('CLASSIFICATION')
        print('C best param')
        print(FH.best_params_['C'])
        print('gamma best param')
        print(FH.best_params_['gamma'])

        FM = SVC(C = FH.best_params_['C'],
                kernel = FH.best_params_['kernel'],
                gamma = FH.best_params_['gamma'])
        FM.fit(FX,FY)

        ####################################################
        ################## TESTING #########################

        print('PREDICTING FRENCH TEST SET')
        FYF = FM.predict(FXT)
        cm = confusion_matrix(FYT,FYF)
        print(cm)
        Ccm_list.append(cm)
        print('PREDICTING CHINESE TEST SET')
        CFYF = FM.predict(CXT)
        cm = confusion_matrix(CYT,CFYF)
        print(cm)
        Fcm_list.append(cm)
        print('PREDICTING MIX TEST SET')
        MCYF = FM.predict(MXT)
        cm = confusion_matrix(MYT,MCYF)
        print(cm)
        Mcm_list.append(cm)


######################################################
###################### RESULTS #######################
print('RESULTS')
def calculate_percentage_confusion_matrix(confusion_matrix_list, tot):
        pcms = []
        for i in confusion_matrix_list:
                true_negative = (i[0,0]/tot)*100
                false_negative = (i[1,0]/tot)*100
                true_positive = (i[1,1]/tot)*100
                false_positive = (i[0,1]/tot)*100
                pcm = np.array([[true_negative , false_positive],[false_negative, true_positive]])
                pcms.append(pcm)
        return pcms

def return_tot_elements(cm):
        tot = cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]
        return tot

def return_statistics_pcm(pcms):
        max_true_negative = 0
        max_false_negative = 0
        max_true_positive = 0
        max_false_positive = 0
        min_true_negative = 100
        min_false_negative = 100
        min_true_positive = 100
        min_false_positive = 100
        count_true_negative = 0
        count_false_negative = 0
        count_true_positive = 0
        count_false_positive = 0
        for i in pcms:
                true_negative = i[0,0]
                false_negative = i[1,0]
                true_positive = i[1,1]
                false_positive = i[0,1]

                count_true_negative += true_negative
                count_false_negative += false_negative
                count_false_positive += false_positive
                count_true_positive += true_positive

                if true_negative > max_true_negative:
                        max_true_negative = true_negative
                if false_negative > max_false_negative:
                        max_false_negative = false_negative
                if true_positive > max_true_positive:
                        max_true_positive = true_positive
                if false_positive > max_false_positive:
                        max_false_positive = false_positive

                if true_negative < min_true_negative:
                        min_true_negative = true_negative
                if false_negative < min_false_negative:
                        min_false_negative = false_negative
                if true_positive < min_true_positive:
                        min_true_positive = true_positive
                if false_positive < min_false_positive:
                        min_false_positive = false_positive
        
        mean_true_negative = count_true_negative/len(pcms)
        mean_false_negative = count_false_negative/len(pcms)
        mean_true_positive = count_true_positive/len(pcms)
        mean_false_positive = count_false_positive/len(pcms)

        mean_matrix = np.array([[mean_true_negative, mean_false_positive],[mean_false_negative, mean_true_positive]])
        max_matrix = np.array([[max_true_negative, max_false_positive],[max_false_negative, max_true_positive]])
        min_matrix = np.array([[min_true_negative, min_false_positive],[min_false_negative, min_true_positive]])

        matrix = []
        matrix.append(mean_matrix)
        matrix.append(max_matrix)
        matrix.append(min_matrix)
        return matrix
                


Ctot = return_tot_elements(Ccm_list[0])
Ccm_list = calculate_percentage_confusion_matrix(Ccm_list, Ctot)


Ftot = return_tot_elements(Fcm_list[0])
Fcm_list = calculate_percentage_confusion_matrix(Fcm_list, Ftot)

Mtot = return_tot_elements(Mcm_list[0])
Mcm_list = calculate_percentage_confusion_matrix(Mcm_list, Mtot)

statistic_C = return_statistics_pcm(Ccm_list)
statistic_F = return_statistics_pcm(Fcm_list)
statistic_M = return_statistics_pcm(Mcm_list)


print(statistic_C)
print(statistic_F)
print(statistic_M)

####################################################################
###################### PLOT IMAGE ##################################
print('PLOT IMAGE')
plt.figure()
plt.imshow(np.reshape(FX[30], (itd.size,itd.size)))
plt.show()