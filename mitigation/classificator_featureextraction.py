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
import torch
from tensorflow.keras.models import Model
import cv2
from tensorflow.keras.preprocessing import image
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
class SVCClassificator:

    def __init__(self, ds_selection = "", kernel= ""):
        self.ds_selection = ds_selection
        self.kernel = kernel

    def evaluate_sigmoid(self,y_pred):
        if y_pred<0.5:
                return 0
        else:
                return 1

    def execute(self):
        #gpus = tf.config.experimental.list_physical_devices('CPU')
        '''gpus = tf.config.experimental.list_physical_devices('GPU')
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
            print('no gpus')'''
        # Confusion matrix lists
        Ccm_list = []
        Fcm_list = []
        Mcm_list = []
        Ccm_1_list = []
        Fcm_1_list = []
        Mcm_1_list = []

        for i in range(30):
                print('CICLE: ' + str(i))
                gpus = tf.config.experimental.list_physical_devices('CPU')
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                        # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
                        try:
                                tf.config.experimental.set_virtual_device_configuration(
                                gpus[0],
                                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1000)])
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

                CXT = itd.CXT
                CYT = itd.CYT
                FXT = itd.FXT
                FYT = itd.FYT
                MXT = itd.MXT
                MYT = itd.MYT

                M = itd.M
                self.size = itd.size

                ###############################################################
                ######################### TESTING #############################
                print('PREDICTING CHINESE TEST SET')
                CYF = []
                CYF_1 = []
                for i in CXT:
                        x = np.reshape(i, (itd.size,itd.size))*255
                        x = cv2.merge([x,x,x])
                        x = image.img_to_array(x)
                        x = np.expand_dims(x, axis=0)
                        feature = M.predict(x, verbose = 0)
                        y_pred = self.evaluate_sigmoid(feature[0][0])
                        y_pred_1 = self.evaluate_sigmoid(feature[1][0])
                        CYF.append(y_pred)
                        CYF_1.append(y_pred_1)
                        
                cm = confusion_matrix(CYT,CYF)
                print(cm)
                cm_1 = confusion_matrix(CYT,CYF_1)
                print(cm_1)
                Ccm_list.append(cm)
                Ccm_1_list.append(cm_1)
                print('Predicting FRENCH TEST SET')
                CFYF = []
                CFYF_1 = []
                for i in FXT:
                        x = np.reshape(i, (itd.size,itd.size))*255
                        x = cv2.merge([x,x,x])
                        x = image.img_to_array(x)
                        x = np.expand_dims(x, axis=0)
                        feature = M.predict(x, verbose = 0)
                        y_pred = self.evaluate_sigmoid(feature[0][0])
                        y_pred_1 = self.evaluate_sigmoid(feature[1][0])
                        CFYF.append(y_pred)
                        CFYF_1.append(y_pred_1)
                        
                cm = confusion_matrix(FYT,CFYF)
                print(cm)
                cm_1 = confusion_matrix(FYT,CFYF_1)
                print(cm_1)
                Fcm_list.append(cm)
                Fcm_1_list.append(cm_1)
                print('PREDICTING MIX TEST SET')
                MYF = []
                MYF_1 = []
                for i in MXT:
                        x = np.reshape(i, (itd.size,itd.size))*255
                        x = cv2.merge([x,x,x])
                        x = image.img_to_array(x)
                        x = np.expand_dims(x, axis=0)
                        feature = M.predict(x, verbose = 0)
                        y_pred = self.evaluate_sigmoid(feature[0][0])
                        y_pred_1 = self.evaluate_sigmoid(feature[1][0])
                        MYF.append(y_pred)
                        MYF_1.append(y_pred_1)
                cm = confusion_matrix(MYT,MYF)
                print(cm)
                Mcm_list.append(cm)
                cm_1 = confusion_matrix(MYT,MYF_1)
                Mcm_1_list.append(cm_1)
                print(cm_1)


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
        Ccm_1_list = calculate_percentage_confusion_matrix(Ccm_1_list, Ctot)

        Ftot = return_tot_elements(Fcm_list[0])
        Fcm_list = calculate_percentage_confusion_matrix(Fcm_list, Ftot)
        Fcm_1_list = calculate_percentage_confusion_matrix(Fcm_1_list, Ftot)

        Mtot = return_tot_elements(Mcm_list[0])
        Mcm_list = calculate_percentage_confusion_matrix(Mcm_list, Mtot)
        Mcm_1_list = calculate_percentage_confusion_matrix(Mcm_1_list, Mtot)

        statistic_C = return_statistics_pcm(Ccm_list)
        statistic_F = return_statistics_pcm(Fcm_list)
        statistic_M = return_statistics_pcm(Mcm_list)
        statistic_C_1 = return_statistics_pcm(Ccm_1_list)
        statistic_F_1 = return_statistics_pcm(Fcm_1_list)
        statistic_M_1 = return_statistics_pcm(Mcm_1_list)

        print('CHINESE')
        print('Exit 0')
        for i in statistic_C:
                print(i)
        print('Exit 1')
        for i in statistic_C_1:
                print(i)
        #print(statistic_C)
        print('FRENCH')
        print('Exit 0')
        for i in statistic_F:
                print(i)
        print('Exit 1')
        for i in statistic_F_1:
                print(i)
        #print(statistic_C)
        print('MIX')
        print('Exit 0')
        for i in statistic_M:
                print(i)
        print('Exit 1')
        for i in statistic_M_1:
                print(i)
        #print(statistic_C)
        


        ####################################################################
        ###################### PLOT IMAGE ##################################
        print('PLOT IMAGE')
        plt.figure()
        plt.imshow(np.reshape(MCX[30], (itd.size,itd.size)))
        plt.show()
