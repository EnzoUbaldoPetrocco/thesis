#! /usr/bin/env python3
import re
from acquisition_evaluation import RModel
from FeatureExtractor_ResNet50 import FeatureExtractor
import cv2
import pathlib
import time
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def get_images(path, n_images):
    images = []
    types = ('*.png', '*.jpg', '*.jpeg')
    paths = []
    for typ in types:
        paths.extend(pathlib.Path(path).glob(typ))
    #sorted_ima = sorted([x for x in paths])
    for i in paths[0: n_images]:
      im = cv2.imread(str(i))
      images.append(im)
    return images

def print_accuracy(cm, n_images):
    acc = cm[0][0]/n_images + cm[1][1]/n_images
    print(acc)
    return acc

def evaluate_sigmoid(y_pred):
        if y_pred<0.5:
                return 0
        else:
                return 1

lambda_grid = [1.00000000e-02, 1.46779927e-02, 2.15443469e-02,  3.16227766e-02,
        4.64158883e-02, 6.81292069e-02, 1.00000000e-01, 1.46779927e-01,
        2.15443469e-01, 3.16227766e-01, 4.64158883e-01, 6.81292069e-01,
        1.00000000e+00, 1.46779927e+00, 2.15443469e+00, 3.16227766e+00,
        4.64158883e+00, 6.81292069e+00, 1.00000000e+01, 1.46779927e+01,
        2.15443469e+01, 3.16227766e+01, 4.64158883e+01, 6.81292069e+01,
        1.00000000e+02]
lamb =  lambda_grid[18]
n_images = 50
chinese_on = get_images("../../TEST/chinese_on", n_images=n_images)
chinese_off = get_images("../../TEST/chinese_off", n_images=n_images)
french_on = get_images("../../TEST/french_on", n_images=n_images)
french_off = get_images("../../TEST/french_off", n_images=n_images)


chinese_model = FeatureExtractor("chinese", lamb=lamb)
french_model = FeatureExtractor("french", lamb=lamb)

chin_rmodel= RModel("chinese", lamb)
fren_rmodel = RModel("french", lamb)

#########################################################
########## STATISTICS ###################################
CXT = chinese_model.CXT
CYT = chinese_model.CYT
FXT = chinese_model.FXT
FYT = chinese_model.FYT
MXT = chinese_model.MXT
MYT = chinese_model.MYT

##########################################################
#################### ORIGINAL MODEL TEST##################
'''print('ORIGINAL MODEL')
print('PREDICTING WITH CHINESE MODEL')
print('predicting CHINESE TEST SET')
CYF = []
CYF_1 = []
for i in CXT:
        x = np.reshape(i, (chinese_model.size,chinese_model.size))*255
        x = cv2.merge([x,x,x])
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        feature = chinese_model.model.predict(x, verbose = 0)
        y_pred = evaluate_sigmoid(feature[0][0])
        y_pred_1 = evaluate_sigmoid(feature[1][0])
        CYF.append(y_pred)
        CYF_1.append(y_pred_1)
       
chinchincm = confusion_matrix(CYT,CYF)
chinchincm_1 = confusion_matrix(CYT,CYF_1)
print('Predicting FRENCH TEST SET')
CFYF = []
CFYF_1 = []
for i in FXT:
        x = np.reshape(i, (chinese_model.size,chinese_model.size))*255
        x = cv2.merge([x,x,x])
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        feature = chinese_model.model.predict(x, verbose = 0)
        y_pred = evaluate_sigmoid(feature[0][0])
        y_pred_1 = evaluate_sigmoid(feature[1][0])
        CFYF.append(y_pred)
        CFYF_1.append(y_pred_1)
      
chinfrencm = confusion_matrix(FYT,CFYF)
chinfrencm_1 = confusion_matrix(FYT,CFYF_1)
print('PREDICTING MIX TEST SET')
MYF = []
MYF_1 = []
for i in MXT:
        x = np.reshape(i, (chinese_model.size,chinese_model.size))*255
        x = cv2.merge([x,x,x])
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        feature = chinese_model.model.predict(x, verbose = 0)
        y_pred = evaluate_sigmoid(feature[0][0])
        y_pred_1 = evaluate_sigmoid(feature[1][0])
        MYF.append(y_pred)
        MYF_1.append(y_pred_1)
chinmixcm = confusion_matrix(MYT,MYF)
chinmixcm_1 = confusion_matrix(MYT,MYF_1)

print('PREDICTING WITH FRENCH MODEL')
print('predicting CHINESE TEST SET')
CYF = []
CYF_1 = []
for i in CXT:
        x = np.reshape(i, (chinese_model.size,chinese_model.size))*255
        x = cv2.merge([x,x,x])
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        feature = french_model.model.predict(x, verbose = 0)
        y_pred = evaluate_sigmoid(feature[0][0])
        y_pred_1 = evaluate_sigmoid(feature[1][0])
        CYF.append(y_pred)
        CYF_1.append(y_pred_1)
  
frenchincm = confusion_matrix(CYT,CYF)
frenchincm_1 = confusion_matrix(CYT,CYF_1)
print('Predicting FRENCH TEST SET')
CFYF = []
CFYF_1 = []
for i in FXT:
        x = np.reshape(i, (chinese_model.size,chinese_model.size))*255
        x = cv2.merge([x,x,x])
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        feature = french_model.model.predict(x, verbose = 0)
        y_pred = evaluate_sigmoid(feature[0][0])
        y_pred_1 = evaluate_sigmoid(feature[1][0])
        CFYF.append(y_pred)
        CFYF_1.append(y_pred_1)
        
frenfrencm = confusion_matrix(FYT,CFYF)
frenfrencm_1 = confusion_matrix(FYT,CFYF_1)
print('PREDICTING MIX TEST SET')
MYF = []
MYF_1 = []
for i in MXT:
        x = np.reshape(i, (chinese_model.size,chinese_model.size))*255
        x = cv2.merge([x,x,x])
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        feature = french_model.model.predict(x, verbose = 0)
        y_pred = evaluate_sigmoid(feature[0][0])
        y_pred_1 = evaluate_sigmoid(feature[1][0])
        MYF.append(y_pred)
        MYF_1.append(y_pred_1)
frenmixcm = confusion_matrix(MYT,MYF)
frenmixcm_1 = confusion_matrix(MYT,MYF_1)

###########################################
############## PRINT RESULTS #############
print('Accuracy of chinese images from chinese model output 0:')
print_accuracy(chinchincm, len(CXT))
print('Accuracy of chinese images from chinese model output 1:')
print_accuracy(chinchincm_1, len(CXT))
print('Accuracy of french images from chinese model output 0:')
print_accuracy(chinfrencm, len(FXT))
print('Accuracy of french images from chinese model output 1:')
print_accuracy(chinfrencm_1, len(FXT))
print('Accuracy of mixed images from chinese model output 0:')
print_accuracy(chinmixcm, len(MXT))
print('Accuracy of mixed images from chinese model output 1:')
print_accuracy(chinmixcm_1, len(MXT))

print('Accuracy of chinese images from french model output 0:')
print_accuracy(frenchincm, len(CXT))
print('Accuracy of chinese images from french model output 1:')
print_accuracy(frenchincm_1, len(CXT))
print('Accuracy of french images from french model output 0:')
print_accuracy(frenfrencm, len(FXT))
print('Accuracy of french images from french model output 1:')
print_accuracy(frenfrencm_1, len(FXT))
print('Accuracy of mixed images from french model output 0:')
print_accuracy(frenmixcm, len(MXT))
print('Accuracy of mixed images from french model output 1:')
print_accuracy(frenmixcm, len(MXT))
'''

###############################################################
######################### TESTING #############################
print('RECREATED MODEL')
print('PREDICTING WITH CHINESE MODEL')
print('predicting CHINESE TEST SET')
CYF = []
CYF_1 = []
for i in CXT:
        x = np.reshape(i, (chinese_model.size,chinese_model.size))*255
        x = cv2.merge([x,x,x])
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        an_y_pred = chin_rmodel.model.predict(x, verbose=0)
        y_pred = chin_rmodel.evaluate_sigmoid(an_y_pred, 0)
        y_pred_1 = chin_rmodel.evaluate_sigmoid(an_y_pred, 1)
        CYF.append(y_pred)
        CYF_1.append(y_pred_1)
        
chinchincm = confusion_matrix(CYT,CYF)
chinchincm_1 = confusion_matrix(CYT,CYF_1)
print('Predicting FRENCH TEST SET')
CFYF = []
CFYF_1 = []
for i in FXT:
        x = np.reshape(i, (chinese_model.size,chinese_model.size))*255
        x = cv2.merge([x,x,x])
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        an_y_pred = chin_rmodel.model.predict(x, verbose=0)
        y_pred = chin_rmodel.evaluate_sigmoid(an_y_pred, 0)
        y_pred_1 = chin_rmodel.evaluate_sigmoid(an_y_pred, 1)
        CFYF.append(y_pred)
        CFYF_1.append(y_pred_1)
        
chinfrencm = confusion_matrix(FYT,CFYF)
chinfrencm_1 = confusion_matrix(FYT,CFYF_1)
print('Predicting MIX TEST SET')
MYF = []
MYF_1 = []
for i in MXT:
        x = np.reshape(i, (chinese_model.size,chinese_model.size))*255
        x = cv2.merge([x,x,x])
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        an_y_pred = chin_rmodel.model.predict(x, verbose=0)
        y_pred = chin_rmodel.evaluate_sigmoid(an_y_pred, 0)
        y_pred_1 = chin_rmodel.evaluate_sigmoid(an_y_pred, 1)
        MYF.append(y_pred)
        MYF_1.append(y_pred_1)
chinmixcm = confusion_matrix(MYT,MYF)
chinmixcm_1 = confusion_matrix(MYT,MYF_1)


print('PREDICTING WITH FRENCH MODEL')
print('predicting CHINESE TEST SET')
CYF = []
CYF_1 = []
for i in CXT:
        x = np.reshape(i, (chinese_model.size,chinese_model.size))*255
        x = cv2.merge([x,x,x])
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        an_y_pred = fren_rmodel.model.predict(x, verbose=0)
        y_pred = fren_rmodel.evaluate_sigmoid(an_y_pred, 0)
        y_pred_1 = fren_rmodel.evaluate_sigmoid(an_y_pred, 1)
        CYF.append(y_pred)
        CYF_1.append(y_pred_1)
        
frenchincm = confusion_matrix(CYT,CYF)
frenchincm_1 = confusion_matrix(CYT,CYF_1)
print('Predicting FRENCH TEST SET')
CFYF = []
CFYF_1 = []
for i in FXT:
        x = np.reshape(i, (chinese_model.size,chinese_model.size))*255
        x = cv2.merge([x,x,x])
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        an_y_pred = fren_rmodel.model.predict(x, verbose=0)
        y_pred = fren_rmodel.evaluate_sigmoid(an_y_pred, 0)
        y_pred_1 = fren_rmodel.evaluate_sigmoid(an_y_pred, 1)
        CFYF.append(y_pred)
        CFYF_1.append(y_pred_1)
        
frenfrencm = confusion_matrix(FYT,CFYF)
frenfrencm_1 = confusion_matrix(FYT,CFYF_1)
print('Predicting MIX TEST SET')
MYF = []
MYF_1 = []
for i in MXT:
        x = np.reshape(i, (chinese_model.size,chinese_model.size))*255
        x = cv2.merge([x,x,x])
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        an_y_pred = fren_rmodel.model.predict(x, verbose=0)
        y_pred = fren_rmodel.evaluate_sigmoid(an_y_pred, 0)
        y_pred_1 = fren_rmodel.evaluate_sigmoid(an_y_pred, 1)
        MYF.append(y_pred)
        MYF_1.append(y_pred_1)
frenmixcm = confusion_matrix(MYT,MYF)
frenmixcm_1 = confusion_matrix(MYT,MYF_1)

###########################################
############## PRINT RESULTS #############
print('Accuracy of chinese images from chinese model output 0:')
print_accuracy(chinchincm, len(CXT))
print('Accuracy of chinese images from chinese model output 1:')
print_accuracy(chinchincm_1, len(CXT))
print('Accuracy of french images from chinese model output 0:')
print_accuracy(chinfrencm, len(FXT))
print('Accuracy of french images from chinese model output 1:')
print_accuracy(chinfrencm_1, len(FXT))
print('Accuracy of mixed images from chinese model output 0:')
print_accuracy(chinmixcm, len(MXT))
print('Accuracy of mixed images from chinese model output 1:')
print_accuracy(chinmixcm_1, len(MXT))

print('Accuracy of chinese images from french model output 0:')
print_accuracy(frenchincm, len(CXT))
print('Accuracy of chinese images from french model output 1:')
print_accuracy(frenchincm_1, len(CXT))
print('Accuracy of french images from french model output 0:')
print_accuracy(frenfrencm, len(FXT))
print('Accuracy of french images from french model output 1:')
print_accuracy(frenfrencm_1, len(FXT))
print('Accuracy of mixed images from french model output 0:')
print_accuracy(frenmixcm, len(MXT))
print('Accuracy of mixed images from french model output 1:')
print_accuracy(frenmixcm, len(MXT))



#########################################
########### REAL IMAGES #################

chin_on_from_chin = []
chin_on_from_fren = []
chin_off_from_chin = []
chin_off_from_fren = []

fren_on_from_chin = []
fren_on_from_fren = []
fren_off_from_chin = []
fren_off_from_fren = []

##########################################
####### HERE I TAKE THE TIME ############

start = time.time()
print('OUT 0')
for i in chinese_on:
    res = chin_rmodel.evaluate_image(i,0,1)
    #print(res)
    chin_on_from_chin.append(res)  
    res = fren_rmodel.evaluate_image(i,0,1)
    #print(res)   
    chin_on_from_fren.append(res)  
for i in chinese_off:
    res = chin_rmodel.evaluate_image(i,0,0)
    #print(res)
    chin_off_from_chin.append(res) 
    res = fren_rmodel.evaluate_image(i,0,0)
    #print(res)
    chin_off_from_fren.append(res)     

for i in french_on:
    res = chin_rmodel.evaluate_image(i,0,1)
    #print(res)
    fren_on_from_chin.append(res) 
    res = fren_rmodel.evaluate_image(i,0,1)
    #print(res)
    fren_on_from_fren.append(res) 
for i in french_off:
    res = chin_rmodel.evaluate_image(i,0,0)
    #print(res)
    fren_off_from_chin.append(res)
    res = fren_rmodel.evaluate_image(i,0,0)
    #print(res)
    fren_off_from_fren.append(res) 



##########################################################
################## CONFUSION MATRICES ###################

y_pred = chin_on_from_chin + chin_off_from_chin 
y_true = list(np.ones(len(chin_on_from_chin))) +  list(np.zeros(len(chin_off_from_chin)))
chinchincm = confusion_matrix(y_true, y_pred)

y_pred = fren_on_from_chin + fren_off_from_chin
y_true =  list(np.ones(len(fren_on_from_chin))) +  list(np.zeros(len(fren_off_from_chin)))
chinfrencm = confusion_matrix(y_true, y_pred)

y_pred = chin_on_from_fren + chin_off_from_fren
y_true = list(np.ones(len(chin_on_from_fren))) + list(np.zeros(len(chin_off_from_fren)))
frenchincm = confusion_matrix(y_true, y_pred)

y_pred = fren_on_from_fren + fren_off_from_fren
y_true = list(np.ones(len(fren_on_from_fren))) +  list(np.zeros(len(fren_off_from_fren)))
frenfrencm = confusion_matrix(y_true, y_pred)

############################################################
################ PRINT RESULTS ###########################

print('Accuracy of chinese images from chinese model output 0:')
print_accuracy(chinchincm, len(y_pred))
print('Accuracy of french images from chinese model output 0:')
print_accuracy(chinfrencm, len(y_pred))
print('Accuracy of chinese images from french model output 0:')
print_accuracy(frenchincm, len(y_pred))
print('Accuracy of french images from french model output 0:')
print_accuracy(frenfrencm, len(y_pred))

##########################################################
################### OUT 1 ##############################
print('OUT 1')
chin_on_from_chin = []
chin_on_from_fren = []
chin_off_from_chin = []
chin_off_from_fren = []

fren_on_from_chin = []
fren_on_from_fren = []
fren_off_from_chin = []
fren_off_from_fren = []

for i in chinese_on:
    res = chin_rmodel.evaluate_image(i,1,1)
    #print(res)
    chin_on_from_chin.append(res)  
    res = fren_rmodel.evaluate_image(i,1,1)
    #print(res)   
    chin_on_from_fren.append(res)  
for i in chinese_off:
    res = chin_rmodel.evaluate_image(i,1,0)
    #print(res)
    chin_off_from_chin.append(res) 
    res = fren_rmodel.evaluate_image(i,1,0)
    #print(res)
    chin_off_from_fren.append(res)   

for i in french_on:
    res = chin_rmodel.evaluate_image(i,1,1)
    #print(res)
    fren_on_from_chin.append(res) 
    res = fren_rmodel.evaluate_image(i,1,1)
    #print(res)
    fren_on_from_fren.append(res) 
for i in french_off:
    res = chin_rmodel.evaluate_image(i,1,0)
    #print(res)
    fren_off_from_chin.append(res)
    res = fren_rmodel.evaluate_image(i,1,0)
    #print(res)
    fren_off_from_fren.append(res) 

##########################################################
################## CONFUSION MATRICES ###################

y_pred = chin_on_from_chin + chin_off_from_chin 
y_true = list(np.ones(len(chin_on_from_chin))) + list(np.zeros(len(chin_off_from_chin)))
chinchincm = confusion_matrix(y_true, y_pred)

y_pred = fren_on_from_chin + fren_off_from_chin
y_true =  list(np.ones(len(fren_on_from_chin))) +  list(np.zeros(len(fren_off_from_chin)))
chinfrencm = confusion_matrix(y_true, y_pred)

y_pred = chin_on_from_fren + chin_off_from_fren
y_true = list(np.ones(len(chin_on_from_fren))) + list(np.zeros(len(chin_off_from_fren)))
frenchincm = confusion_matrix(y_true, y_pred)

y_pred = fren_on_from_fren + fren_off_from_fren
y_true = list(np.ones(len(fren_on_from_fren))) +  list(np.zeros(len(fren_off_from_fren)))
frenfrencm = confusion_matrix(y_true, y_pred)

############################################################
##################### PRINT RESULTS ########################
print('Accuracy of chinese images from chinese model output 1:')
print_accuracy(chinchincm, len(y_pred))
print('Accuracy of french images from chinese model output 1:')
print_accuracy(chinfrencm, len(y_pred))

print('Accuracy of chinese images from french model output 1:')
print_accuracy(frenchincm, len(y_pred))
print('Accuracy of french images from french model output 1:')
print_accuracy(frenfrencm, len(y_pred))

##########################################################


end = time.time()
print('Time is:', str(end-start))

#############################
## ALERT ###################
plt.figure()
plt.imshow(np.reshape(CXT[0], (chin_rmodel.itd.size, chin_rmodel.itd.size)))
plt.show()