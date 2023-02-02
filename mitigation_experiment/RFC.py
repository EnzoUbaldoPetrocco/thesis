#! /usr/bin/env python3

from FeatureExtractor_ResNet50 import FeatureExtractor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import manipulating_images_better
from math import floor
from sklearn.metrics import confusion_matrix
import tensorflow as tf

class RFCClassificator:

    def __init__(self, ds_selection = ""):
        self.ds_selection = ds_selection

    def execute(self):
        # Confusion matrix lists
        Ccm_list = []
        Fcm_list = []
        Mcm_list = []

        for i in range(30):
                print('CICLE: ' + str(i))

                gpus = tf.config.experimental.list_physical_devices('CPU')
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                        # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
                        try:
                                tf.config.experimental.set_virtual_device_configuration(
                                gpus[0],
                                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])
                                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                        except RuntimeError as e:
                                # Virtual devices must be set before GPUs have been initialized
                                print(e)
                else:
                        print('no gpus')
                ############################################################
                ############### READ DATA ##################################
                itd = FeatureExtractor(self.ds_selection)
                proportion = 1.5

                CX = itd.CX[0:int(len(itd.CX)/proportion)]
                CY = itd.CY[0:int(len(itd.CX)/proportion)]
                FX = itd.FX[0:int(len(itd.CX)/proportion)]
                FY = itd.FY[0:int(len(itd.CX)/proportion)]
                MX = itd.MX[0:int(len(itd.CX)/proportion)]
                MY = itd.MY[0:int(len(itd.CX)/proportion)]

                CXT = itd.CX[int(len(itd.CX)/proportion):int(len(itd.CX))-1]
                CYT = itd.CY[int(len(itd.CX)/proportion):int(len(itd.CX))-1]
                FXT = itd.FX[int(len(itd.CX)/proportion):int(len(itd.CX))-1]
                FYT = itd.FY[int(len(itd.CX)/proportion):int(len(itd.CX))-1]
                MXT= itd.MX[int(len(itd.MX)/proportion):int(len(itd.MX))-1]
                MYT = itd.MY[int(len(itd.MX)/proportion):int(len(itd.MX))-1]

                self.size = itd.size


                #####################################################################
                ################### MODEL SELECTION (HYPERPARAMETER TUNING)##########
                print('MODEL SELECTION AND TUNING')

                rfc=RandomForestClassifier(random_state=42)
                logspace_n_estimators = []
                logspace_max_depth = []
                for i in np.logspace(0,2,30):
                        logspace_max_depth.append(int(i))
                for i in np.logspace(0,3,50):
                    logspace_n_estimators.append(int(i))
                param_grid = { 
                    'n_estimators': [300], #logspace_n_estimators,
                    'max_features': [ 'sqrt', 'log2'],
                    'max_depth' : logspace_max_depth,
                    'criterion' :['gini', 'entropy']
                    }
                
                CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
                
                if self.ds_selection == "chinese":
                    CV_rfc.fit(CX, CY)
                if self.ds_selection == "french":
                    CV_rfc.fit(FX, FY)
                if self.ds_selection == "mix":
                    CV_rfc.fit(MX, MY)

                print(CV_rfc.best_params_)

                rfc1=RandomForestClassifier(random_state=42,
                max_features=CV_rfc.best_params_['max_features'], n_estimators= CV_rfc.best_params_['n_estimators'],
                max_depth=CV_rfc.best_params_['max_depth'], criterion=CV_rfc.best_params_['criterion'])

                if self.ds_selection == "chinese":
                    rfc1.fit(CX, CY)
                if self.ds_selection == "french":
                    rfc1.fit(FX, FY)
                if self.ds_selection == "mix":
                    rfc1.fit(MX, MY)
                ####################################################
                ################## TESTING #########################

                print('PREDICTING CHINESE TEST SET')
                CYF = rfc1.predict(CXT)
                cm = confusion_matrix(CYT,CYF)
                print(cm)
                Ccm_list.append(cm)
                print('Predicting FRENCH TEST SET')
                CFYF = rfc1.predict(FXT)
                cm = confusion_matrix(FYT,CFYF)
                print(cm)
                Fcm_list.append(cm)
                print('PREDICTING MIX TEST SET')
                MYF = rfc1.predict(MXT)
                cm = confusion_matrix(MYT,MYF)
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

        ###################################################################
        ################## PRINT RESULTS ##################################
        accuracy_C = statistic_C[0][0][0] + statistic_C[0][1][1]
        accuracy_F = statistic_F[0][0][0] + statistic_F[0][1][1]
        accuracy_M = statistic_M[0][0][0] + statistic_M[0][1][1]
        print('Chinese Accuracy Out 0 ', accuracy_C, '%')
        print('French Accuracy Out 0 ', accuracy_F, '%')
        print('Mixed Accuracy Out 0 ', accuracy_M, '%')



        ####################################################################
        ###################### PLOT IMAGE ##################################
        print('PLOT IMAGE')
        plt.figure()
        plt.imshow(np.reshape(CX[30], (itd.size,itd.size)))
        plt.show()