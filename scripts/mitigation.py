#! /usr/bin/env python3

import pathlib
import random

import cv2
import numpy as np
import tensorflow as tf
from keras import Model, layers, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Flatten, Input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from math import floor

#####################################################################
###                        PARAMETERS                             ###
size = 75
n = 470
ds_selection = 'french'
dl_prop = 0.7
ml_prop = 0.8
lambda_grid = [1.00000000e-02, 1.46779927e-02, 2.15443469e-02,  3.16227766e-02,
        4.64158883e-02, 6.81292069e-02, 1.00000000e-01, 1.46779927e-01,
        2.15443469e-01, 3.16227766e-01, 4.64158883e-01, 6.81292069e-01,
        1.00000000e+00, 1.46779927e+00, 2.15443469e+00, 3.16227766e+00,
        4.64158883e+00, 6.81292069e+00, 1.00000000e+01, 1.46779927e+01,
        2.15443469e+01, 3.16227766e+01, 4.64158883e+01, 6.81292069e+01,
        1.00000000e+02]
lamb = 0
# DL 
batch_size = 1
validation_split = 0.1
ep = 100
verbose_param = 1
lr_reduce = ReduceLROnPlateau(monitor='val_dense_accuracy', factor=0.2, patience=3, verbose=1, mode='max', min_lr=1e-8)
early = EarlyStopping(monitor='val_dense_accuracy', min_delta=0.001, patience=20, verbose=1, mode='auto')
lr_reduce_1 = ReduceLROnPlateau(monitor='val_dense_1_accuracy', factor=0.2, patience=3, verbose=1, mode='max', min_lr=1e-8)
early_1 = EarlyStopping(monitor='val_dense_1_accuracy', min_delta=0.001, patience=20, verbose=1, mode='auto')
learning_rate= 4e-4
adam = optimizers.Adam(learning_rate)
optimizer = adam
model = 0
# ML
points = 70
kernel = 'rbf'
logspaceC = np.logspace(-2,2.5,points)
logspaceGamma = np.logspace(-2,2.5,points)
grid = {'C':        logspaceC,
        'kernel':   [kernel],
        'gamma':    logspaceGamma}
MS = GridSearchCV(estimator = SVC(),
                                param_grid = grid,
                                scoring = 'balanced_accuracy',
                                cv = 10,
                                verbose = 0)  
#####################################################################
###                        FUNCTIONS                              ### 
def acquire_images(path):
    images = []
    types = ('*.png', '*.jpg', '*.jpeg')
    paths = []
    for typ in types:
        paths.extend(pathlib.Path(path).glob(typ))
    #sorted_ima = sorted([x for x in paths])
    for i in paths:
        im = cv2.imread(str(i))
        #im = cv2.merge([im,im,im])
        im = image.img_to_array(im)
        im = np.expand_dims(im, axis=0)
        images.append(im)
    return images

def mix(list):
    for i in range(9999):
        index = random.randint(0,len(list)-1)
        temp = list[index]
        list.pop(index)
        list.append(temp)
    return list

def mix_ds(X, y):
    for i in range(9999):
        index = random.randint(0,len(X)-1)
        temp = X[index]
        X.pop(index)
        X.append(temp)
        temp_y = y[index]
        y.pop(index)
        y.append(temp_y)
    return X, y

def create_labels(list, labels, times):
    for i in range(times):
        list.append(tf.constant(labels))
    return list

def custom_loss_w1(y_true,y_pred):
    # Calculate lambda * ||Wc - Wf||^2
    weights1 = model.layers[len(model.layers)-1].kernel
    weights2 = model.layers[len(model.layers)-2].kernel
    dist = tf.norm(weights1-weights2, ord='euclidean')
    dist2 = tf.multiply(tf.multiply(dist,dist) , lamb)
    # Loss
    loss = tf.keras.losses.binary_crossentropy(y_true[0][1], y_pred[0])
    mask = tf.math.multiply(0.5, tf.math.add((tf.math.add(y_true[0][0], 0.0)), tf.math.abs(tf.math.subtract(y_true[0][0], 0.0))))     
    res = tf.math.add(loss , dist2)
    if mask > 0 :
        return res
    else:
            return 0.0

def custom_loss_w2(y_true,y_pred):
    # Calculate lambda * ||Wc - Wf||^2
    weights1 = model.layers[len(model.layers)-1].kernel
    weights2 = model.layers[len(model.layers)-2].kernel
    dist = tf.norm(weights1-weights2, ord='euclidean')
    dist2 = tf.multiply(tf.multiply(dist,dist) , lamb)
    # Loss
    loss = tf.keras.losses.binary_crossentropy(y_true[0][1], y_pred[0])
    mask = tf.math.multiply(0.5, tf.math.add((tf.math.add(y_true[0][0], 0)),tf.math.abs(tf.math.subtract(y_true[0][0], 0))))  ##.5*( a + b + |a - b |)
    res = tf.math.add(loss , dist2)
    #return res
    if mask > 0 :
        return 0.0
    else:
        return res

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
                        
#####################################################################
###                        PREPARING FOR CICLES                   ###
chinese_off_temp = acquire_images('../../' + str(size) + '/cinesi')
chinese_on_temp = acquire_images('../../' + str(size) + '/cinesi accese')
french_off_temp = acquire_images('../../' + str(size) + '/francesi')
french_on_temp = acquire_images('../../' + str(size) + '/francesi accese')

chinese_off = []
chinese_on = []
french_off = []
french_on = []
for im in chinese_off_temp:
    chinese_off.append(tf.constant(im[0]))
for im in chinese_on_temp:
    chinese_on.append(tf.constant(im[0]))
for im in french_off_temp:
    french_off.append(tf.constant(im[0]))
for im in french_off_temp:
    french_on.append(tf.constant(im[0]))

Ccm_list = []
Fcm_list = []
Ccm_1_list = []
Fcm_1_list = []

#####################################################################
###                        CICLE                                  ###
for i in range(30):
    chinese_off = mix(chinese_off)
    chinese_on = mix(chinese_on)
    french_off = mix(french_off)
    french_on = mix(french_on)

#####################################################################
###                        DEEP LEARNING                          ###

    X_dl = []
    y_dl = []

    if ds_selection == 'french':
        X_dl = chinese_off[0:floor(n*dl_prop*0.1)]
        X_dl = X_dl + french_off[0: floor(n*dl_prop)]
        X_dl = X_dl + chinese_on[0: floor(n*dl_prop*0.1)]
        X_dl = X_dl + french_on[0: floor(n*dl_prop)]
        y_dl = create_labels(y_dl, [0.0, 0.0], floor(n*dl_prop*0.1))
        y_dl = create_labels(y_dl, [0.0, 1.0], floor(n*dl_prop))
        y_dl = create_labels(y_dl, [1.0, 0.0], floor(n*dl_prop*0.1))
        y_dl = create_labels(y_dl, [1.0, 1.0], floor(n*dl_prop))

        
        
    if ds_selection == 'chinese':
        X_dl = chinese_off[0: floor(n*dl_prop)]
        X_dl = X_dl + french_off[0: floor(n*dl_prop*0.1)]
        X_dl = X_dl + chinese_on[0: floor(n*dl_prop)]
        X_dl = X_dl + french_on[0: floor(n*dl_prop*0.1)]
        y_dl = create_labels(y_dl, [0.0, 0.0], floor(n*dl_prop))
        y_dl = create_labels(y_dl, [0.0, 1.0], floor(n*dl_prop*0.1))
        y_dl = create_labels(y_dl, [1.0, 0.0], floor(n*dl_prop))
        y_dl = create_labels(y_dl, [1.0, 1.0], floor(n*dl_prop*0.1))

    Cy_ml = []
    Fy_ml = []
    CX_ml = chinese_off[floor(n*dl_prop): n-1]
    FX_ml = french_off[floor(n*dl_prop): n-1]
    CX_ml = CX_ml + chinese_on[floor(n*dl_prop): n-1]
    FX_ml = FX_ml + french_on[floor(n*dl_prop): n-1]
    Cy_ml = create_labels(Cy_ml, [0], floor(n*(1-dl_prop))-1)
    Fy_ml = create_labels(Fy_ml, [0], floor(n*(1-dl_prop))-1)
    Cy_ml = create_labels(Cy_ml, [0], floor(n*(1-dl_prop))-1)
    Fy_ml = create_labels(Fy_ml, [0], floor(n*(1-dl_prop))-1)
 
    print(f"Shape of X_dl: {np.shape(X_dl)}")
    print(f"Shape of y_dl: {np.shape(y_dl)}")
    print(f"Shape of CX_ml: {np.shape(CX_ml)}")
    print(f"Shape of Cy_ml: {np.shape(Cy_ml)}")
    print(f"Shape of FX_ml: {np.shape(FX_ml)}")
    print(f"Shape of Fy_ml: {np.shape(Fy_ml)}")


    X_dl, y_dl = mix_ds(X_dl, y_dl)
    CX_ml, Cy_ml = mix_ds(CX_ml, Cy_ml)
    FX_ml, Fy_ml = mix_ds(FX_ml, Fy_ml)

    input = Input((size, size, 3))
    x = ResNet50( #tf.keras.applications.efficientnet.EfficientNetB0
        input_shape=(size, size, 3),
        weights='imagenet',
        include_top=False)(input)
    #model.summary()
    x = Flatten()(x)
    chin = Dropout(0.15)(x)
    fren = Dropout(0.15)(x)
    chin = Dense(50, activation = 'relu', name='output')(chin)
    fren = Dense(50, activation = 'relu', name='output_1')(fren)
    chin = Dense(1, activation='sigmoid', name='dense')(chin)
    fren = Dense(1, activation='sigmoid', name='dense_1')(fren)
    model = Model(inputs=input,
                    outputs = [chin,fren],
                    name= 'model')
    #model.summary()
    model.trainable = True
    for layer in model.layers[1].layers:
        layer.trainable = False
    #model.summary()
    for layer in model.layers[1].layers[-1:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    model.compile(loss=[custom_loss_w1, custom_loss_w2], optimizer=optimizer, metrics=["accuracy"])

    train_size = int(len(X_dl)*0.9)
    X_list = X_dl[0:train_size]
    y_list = y_dl[0:train_size]
    X_val_list = X_dl[train_size:len(X_dl)-1]
    y_val_list = y_dl[train_size:len(X_dl)-1]
    X = tf.stack(X_list)
    y = tf.stack(y_list)
    X_val = tf.stack(X_val_list)
    y_val = tf.stack(y_val_list)
    history = model.fit(X , y,
            epochs=ep, validation_data=(X_val, y_val), 
            callbacks=[early, lr_reduce],verbose=verbose_param, batch_size=batch_size)

    chin_model = Model(inputs=model.inputs, outputs= model.get_layer('output').output)
    fren_model = Model(inputs=model.inputs, outputs= model.get_layer('output_1').output) 


    chin_features = []
    fren_features = []
    for x in CX_ml:
        chin_feature = chin_model.predict(np.expand_dims(x, axis=0),verbose= 0)
        fren_feature = fren_model.predict(np.expand_dims(x, axis=0),verbose= 0)
        chin_features.append(chin_feature[0])
        fren_features.append(fren_feature[0])

    
    CX_chin = chin_features
    CX_fren = fren_features

    chin_features = []
    fren_features = []
    for x in FX_ml:
        chin_feature = chin_model.predict(np.expand_dims(x, axis=0),verbose= 0)
        fren_feature = fren_model.predict(np.expand_dims(x, axis=0),verbose= 0)
        chin_features.append(chin_feature[0])
        fren_features.append(fren_feature[0])

    FX_chin = chin_features
    FX_fren = fren_features

    CXT_chin = CX_chin[int(len(CX_chin)*ml_prop): int(len(CX_chin))-1]
    CXT_fren = CX_fren[int(len(CX_fren)*ml_prop): int(len(CX_fren))-1]
    CyT_chin = Cy_ml[int(len(Cy_ml)*ml_prop): int(len(Cy_ml))-1]
    CyT_fren = CyT_chin

    FXT_chin = FX_chin[int(len(FX_chin)*ml_prop): int(len(FX_chin))-1]
    FXT_fren = FX_fren[int(len(FX_fren)*ml_prop): int(len(FX_fren))-1]
    FyT_chin = Fy_ml[int(len(Fy_ml)*ml_prop): int(len(Fy_ml))-1]
    FyT_fren = FyT_chin

    CX_chin = CX_chin[0:int(len(CX_chin)*ml_prop)]
    CX_fren = CX_fren[0:int(len(CX_fren)*ml_prop)]
    Cy_chin = Cy_ml[0:int(len(Cy_ml)*ml_prop)]
    Cy_fren = CyT_chin

    FX_chin = FX_chin[0:int(len(FX_chin)*ml_prop)]
    FX_fren = FX_fren[0:int(len(FX_fren)*ml_prop)]
    Fy_chin = Fy_ml[0:int(len(Fy_ml)*ml_prop)]
    Fy_fren = FyT_chin

    print('USING CHINESE OUTPUT ...')
    if ds_selection == 'chinese':
        H_chin = MS.fit(CX_chin, Cy_chin)
    if ds_selection == 'french':
        H_chin = MS.fit(FX_chin, Fy_chin)
    print('C best param')
    print(H_chin.best_params_['C'])
    print('gamma best param')
    print(H_chin.best_params_['gamma'])

    M_chin = SVC(C = H_chin.best_params_['C'],
            kernel = H_chin.best_params_['kernel'],
            gamma = H_chin.best_params_['gamma'])

    if ds_selection == "chinese":
        M_chin = MS.fit(CX_chin, Cy_chin)
    if ds_selection == "french":
        M_chin = MS.fit(FX_chin, Fy_chin)

    print('PREDICTING CHINESE TEST SET')
    CYF = M_chin.predict(CXT_chin)
    cm = confusion_matrix(CyT_chin,CYF)
    print(cm)
    Ccm_list.append(cm)
    print('Predicting FRENCH TEST SET')
    CFYF = M_chin.predict(FXT_chin)
    cm = confusion_matrix(FyT_chin,CFYF)
    print(cm)
    Fcm_list.append(cm)

    print('USING FRENCH OUTPUT ...')
    if ds_selection == 'chinese':
        H_fren = MS.fit(CX_fren, Cy_fren)
    if ds_selection == 'french':
        H_fren = MS.fit(FX_fren, Fy_fren)
    print('C best param')
    print(H_fren.best_params_['C'])
    print('gamma best param')
    print(H_fren.best_params_['gamma'])

    M_fren = SVC(C = H_fren.best_params_['C'],
            kernel = H_fren.best_params_['kernel'],
            gamma = H_fren.best_params_['gamma'])

    if ds_selection == "chinese":
        M_fren = MS.fit(CX_fren, Cy_fren)
    if ds_selection == "french":
        M_fren = MS.fit(FX_fren, Fy_fren)

    print('PREDICTING CHINESE TEST SET')
    CYF = M_fren.predict(CXT_fren)
    cm = confusion_matrix(CyT_fren,CYF)
    print(cm)
    Ccm_1_list.append(cm)
    print('Predicting FRENCH TEST SET')
    CFYF = M_fren.predict(FXT_fren)
    cm = confusion_matrix(FyT_fren,CFYF)
    print(cm)
    Fcm_1_list.append(cm)

Ctot = return_tot_elements(Ccm_list[0])
Ccm_list = calculate_percentage_confusion_matrix(Ccm_list, Ctot)
Ccm_1_list = calculate_percentage_confusion_matrix(Ccm_1_list, Ctot)

Ftot = return_tot_elements(Fcm_list[0])
Fcm_list = calculate_percentage_confusion_matrix(Fcm_list, Ftot)
Fcm_1_list = calculate_percentage_confusion_matrix(Fcm_1_list, Ftot)

statistic_C = return_statistics_pcm(Ccm_list)
statistic_F = return_statistics_pcm(Fcm_list)
statistic_C_1 = return_statistics_pcm(Ccm_1_list)
statistic_F_1 = return_statistics_pcm(Fcm_1_list)

print('CHINESE')
print('Exit 0')
for i in statistic_C:
        print(i)
print('Exit 1')
for i in statistic_C_1:
        print(i)
print('FRENCH')
print('Exit 0')
for i in statistic_F:
        print(i)
print('Exit 1')
for i in statistic_F_1:
        print(i)

###################################################################
################## PRINT RESULTS ##################################
accuracy_C = statistic_C[0][0][0] + statistic_C[0][1][1]
accuracy_C_1 = statistic_C_1[0][0][0] + statistic_C_1[0][1][1]
accuracy_F = statistic_F[0][0][0] + statistic_F[0][1][1]
accuracy_F_1 = statistic_F_1[0][0][0] + statistic_F_1[0][1][1]
print('Chinese Accuracy Out 0 ', accuracy_C, '%')
print('Chinese Accuracy Out 1 ', accuracy_C_1, '%')
print('French Accuracy Out 0 ', accuracy_F, '%')
print('French Accuracy Out 1 ', accuracy_F_1, '%')