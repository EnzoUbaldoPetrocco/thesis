#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from FeatureExtractor_ResNet50 import FeatureExtractor
from sklearn.metrics import confusion_matrix
import tensorflow as tf
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

                out0MCX = itd.out0MCX
                out0CXT = itd.out0CXT
                out0MCY = itd.out0MCY
                out0CYT = itd.out0CYT

                out0MFX = itd.out0MFX
                out0FXT = itd.out0FXT
                out0MFY = itd.out0MFY
                out0FYT = itd.out0FYT

                out0MXT = itd.out0MXT
                out0MYT = itd.out0MYT

                out1MCX = itd.out1MCX
                out1CXT = itd.out1CXT
                out1MCY = itd.out1MCY
                out1CYT = itd.out1CYT

                out1MFX = itd.out1MFX
                out1FXT = itd.out1FXT
                out1MFY = itd.out1MFY
                out1FYT = itd.out1FYT

                out1MXT = itd.out1MXT
                out1MYT = itd.out1MYT
                
                
                self.size = itd.size

                ###############################################################
                ######################## TESTING out0 #############################
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
                    H = MS.fit(out0MCX,out0MCY)
                if self.ds_selection == "french":
                    H = MS.fit(out0MFX,out0MFY)
                
                print('CLASSIFICATION')
                print('C best param')
                print(H.best_params_['C'])
                print('gamma best param')
                print(H.best_params_['gamma'])

                out0M = SVC(C = H.best_params_['C'],
                        kernel = H.best_params_['kernel'],
                        gamma = H.best_params_['gamma'])

                if self.ds_selection == "chinese":
                    out0M = MS.fit(out0MCX,out0MCY)
                if self.ds_selection == "french":
                    out0M = MS.fit(out0MFX,out0MFY)


                ######################## TESTING out1 #############################
                points = 70
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
                    H = MS.fit(out1MCX,out1MCY)
                if self.ds_selection == "french":
                    H = MS.fit(out1MFX,out1MFY)
                
                print('CLASSIFICATION')
                print('C best param')
                print(H.best_params_['C'])
                print('gamma best param')
                print(H.best_params_['gamma'])

                out1M = SVC(C = H.best_params_['C'],
                        kernel = H.best_params_['kernel'],
                        gamma = H.best_params_['gamma'])

                if self.ds_selection == "chinese":
                    out1M = MS.fit(out1MCX,out1MCY)
                if self.ds_selection == "french":
                    out1M = MS.fit(out1MFX,out1MFY)

                
                ####################################################
                ################## TESTING out0 #########################

                print('PREDICTING CHINESE TEST SET')
                CYF = out0M.predict(out0CXT)
                cm = confusion_matrix(out0CYT,CYF)
                print(cm)
                Ccm_list.append(cm)
                print('Predicting FRENCH TEST SET')
                CFYF = out0M.predict(out0FXT)
                cm = confusion_matrix(out0FYT,CFYF)
                print(cm)
                Fcm_list.append(cm)
                print('PREDICTING MIX TEST SET')
                MYF = out0M.predict(out0MXT)
                cm = confusion_matrix(out0MYT,MYF)
                print(cm)
                Mcm_list.append(cm)

                ################## TESTING out1 #########################

                print('PREDICTING CHINESE TEST SET')
                CYF = out1M.predict(out1CXT)
                cm = confusion_matrix(out1CYT,CYF)
                print(cm)
                Ccm_1_list.append(cm)
                print('Predicting FRENCH TEST SET')
                CFYF = out1M.predict(out1FXT)
                cm = confusion_matrix(out1FYT,CFYF)
                print(cm)
                Fcm_1_list.append(cm)
                print('PREDICTING MIX TEST SET')
                MYF = out1M.predict(out1MXT)
                cm = confusion_matrix(out1MYT,MYF)
                print(cm)
                Mcm_1_list.append(cm)


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
        
        ###################################################################
        ################## PRINT RESULTS ##################################
        accuracy_C = statistic_C[0][0][0] + statistic_C[0][1][1]
        accuracy_C_1 = statistic_C_1[0][0][0] + statistic_C_1[0][1][1]
        accuracy_F = statistic_F[0][0][0] + statistic_F[0][1][1]
        accuracy_F_1 = statistic_F_1[0][0][0] + statistic_F_1[0][1][1]
        accuracy_M = statistic_M[0][0][0] + statistic_M[0][1][1]
        accuracy_M_1 = statistic_M_1[0][0][0] + statistic_M_1[0][1][1]
        print('Chinese Accuracy Out 0 ', accuracy_C, '%')
        print('Chinese Accuracy Out 1 ', accuracy_C_1, '%')
        print('French Accuracy Out 0 ', accuracy_F, '%')
        print('French Accuracy Out 1 ', accuracy_F_1, '%')
        print('Mixed Accuracy Out 0 ', accuracy_M, '%')
        print('Mixed Accuracy Out 1 ', accuracy_M_1, '%')

        ####################################################################
        ###################### PLOT IMAGE ##################################
        print('PLOT IMAGE')
        plt.figure()
        plt.imshow(np.reshape(itd.CXT[10], (itd.size,itd.size)))
        plt.show()
