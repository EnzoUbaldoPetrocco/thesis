#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from torch import logspace
from FeatureExtractor_EfficientNet import FeatureExtractor
from math import floor
from sklearn.metrics import confusion_matrix
import tensorflow as tf
'''config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)'''
import torch
import os
#CUDA_VISIBLE_DEVICES=""
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class SVCClassificator:

    def __init__(self, ds_selection = "", kernel= ""):
        self.ds_selection = ds_selection
        self.kernel = kernel

    def execute(self):

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        print('Using device:' , device)
        gpus = tf.config.experimental.list_physical_devices('CPU')
        
        '''gpus = tf.config.experimental.list_physical_devices('GPU')
        for device in gpus:
                tf.config.experimental.set_memory_growth(device, True)
        if gpus:
        # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
        else:
            print('no gpus')'''
        # Confusion matrix lists
        Ccm_list = []
        Fcm_list = []
        Mcm_list = []

        for i in range(50):
                print('CICLE: ' + str(i))

                ############################################################
                ############### READ DATA ##################################
                itd = FeatureExtractor(self.ds_selection)

                CX = itd.CX
                CXT = itd.CXT
                CY = itd.CY
                CYT = itd.CYT

                FX = itd.FX
                FXT = itd.FXT
                FY = itd.FY
                FYT = itd.FYT


                MX = itd.MX
                MXT = itd.MXT
                MY = itd.MY
                MYT = itd.MYT


                #####################################################################
                ################### MODEL SELECTION (HYPERPARAMETER TUNING)##########
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print('Using device:' , device)
                points = 80
                print('MODEL SELECTION AND TUNING')
                if self.kernel == 'rbf':
                    logspaceC = np.logspace(-2,2.5,points)
                    logspaceGamma = np.logspace(-2,2.5,points)
                if self.kernel == 'linear':
                    logspaceC = np.logspace(-2,2.5,points)
                    logspaceGamma = np.logspace(-2,2.5,points)
                grid = {'C':        logspaceC,
                        'kernel':   [self.kernel],
                        'gamma':    logspaceGamma}
                MS = GridSearchCV(estimator = SVC(),
                                param_grid = grid,
                                scoring = 'balanced_accuracy',
                                cv = 10,
                                verbose = 0)


                if self.ds_selection == "chinese":
                    H = MS.fit(CX,CY)
                if self.ds_selection == "french":
                    H = MS.fit(FX,FY)
                if self.ds_selection == "mix":
                    H = MS.fit(MX,MY)          
                
                print('CLASSIFICATION')
                print('C best param')
                print(H.best_params_['C'])
                print('gamma best param')
                print(H.best_params_['gamma'])

                M = SVC(C = H.best_params_['C'],
                        kernel = H.best_params_['kernel'],
                        gamma = H.best_params_['gamma'])

                if self.ds_selection == "chinese":
                    M = MS.fit(CX,CY)
                if self.ds_selection == "french":
                    M = MS.fit(FX,FY)
                if self.ds_selection == "mix":
                    M = MS.fit(MX,MY)

                
                ####################################################
                ################## TESTING #########################
                
                print('PREDICTING CHINESE TEST SET')
                CYF = M.predict(CXT)
                cm = confusion_matrix(CYT,CYF)
                print(cm)
                Ccm_list.append(cm)
                print('Predicting FRENCH TEST SET')
                CFYF = M.predict(FXT)
                cm = confusion_matrix(FYT,CFYF)
                print(cm)
                Fcm_list.append(cm)
                print('PREDICTING MIX TEST SET')
                MYF = M.predict(MXT)
                cm = confusion_matrix(MYT,MYF)
                print(cm)
                Mcm_list.append(cm)
                device = torch.device('cpu')  

                


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

        print('CHINESE')
        for i in statistic_C:
                print(i)
        #print(statistic_C)
        print('FRENCH')
        for i in statistic_F:
                print(i)
        #print(statistic_F)
        print('MIX')
        for i in statistic_M:
                print(i)
        #print(statistic_M)


        ####################################################################
        ###################### PLOT IMAGE ##################################
        print('PLOT IMAGE')
        plt.figure()
        plt.imshow(np.reshape(CX[30], (itd.size,itd.size)))
        plt.show()
