#! /usr/bin/env python3
import re
import manipulating_images_better_v2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, Model, optimizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import tensorflow as tf
from skimage.color import gray2rgb
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import keras.layers as L
from keras.layers import Input, Lambda, Dense, Flatten,Dropout, MaxPooling3D
from keras.models import Sequential
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from keras import backend as K
from pathlib import Path
from sklearn.utils import shuffle

'''config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)'''
import torch

import pandas as pd

working_directory = 'MITIGATION'
model = 0
BATCH_SIZE = 1
lambda_grid = [1.00000000e-02, 1.46779927e-02, 2.15443469e-02,  3.16227766e-02,
 4.64158883e-02, 6.81292069e-02, 1.00000000e-01, 1.46779927e-01,
 2.15443469e-01, 3.16227766e-01, 4.64158883e-01, 6.81292069e-01,
 1.00000000e+00, 1.46779927e+00, 2.15443469e+00, 3.16227766e+00,
 4.64158883e+00, 6.81292069e+00, 1.00000000e+01, 1.46779927e+01,
 2.15443469e+01, 3.16227766e+01, 4.64158883e+01, 6.81292069e+01,
 1.00000000e+02]
lamb = 0 #lambda_grid[2]
# out2 dist2 + loss

def unfreeze_model(model, layers_n):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-layers_n:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image

def batch_generator(X, Y, batch_size = BATCH_SIZE):
    indices = np.arange(len(X)) 
    batch=[]
    while True:
            # it might be a good idea to shuffle your data before each epoch
            np.random.shuffle(indices) 
            for i in indices:
                batch.append(i)
                if len(batch)==batch_size:
                    yield X[batch], Y[batch]
                    batch=[]
   
    
#@tf.function


def dummy_loss(y_true, y_pred):
    return 0.0


class FeatureExtractor:

    def custom_loss_w1(self,y_true,y_pred):
        # Calculate lambda * ||Wc - Wf||^2
        w1 = K.get_value(self.model.layers[len(self.model.layers)-1].kernel)
        w2 = K.get_value(self.model.layers[len(self.model.layers)-2].kernel)
        weights1 = np.array([i[0] for i in w1])
        weights2 = np.array([i[0] for i in w2])
        dist = np.linalg.norm(weights1-weights2)
        dist2 = lamb*dist*dist
        # Loss
        loss = tf.keras.losses.binary_crossentropy(y_true[0], y_pred[0])
        loss = tf.cast(loss, dtype=tf.float32)
        dist2 = tf.constant(dist2, dtype=tf.float32)
        mask = K.greater(y_true[0][0], 0)
        res = tf.math.add(loss , dist2)
        #return res
        if mask:
            return res
        else:
            return 0.0

    #@tf.function
    def custom_loss_w2(self, y_true,y_pred):
        # Calculate lambda * ||Wc - Wf||^2
        w1 = K.get_value(self.model.layers[len(self.model.layers)-1].kernel)
        w2 = K.get_value(self.model.layers[len(self.model.layers)-2].kernel)
        weights1 = np.array([i[0] for i in w1])
        weights2 = np.array([i[0] for i in w2])
        dist = np.linalg.norm(weights1-weights2)
        dist2 = lamb*dist*dist
        dist2 = tf.constant(dist2, dtype=tf.float32)
        # Loss
        loss = tf.keras.losses.binary_crossentropy(y_true[0], y_pred[0])
        loss = tf.cast(loss, dtype=tf.float32)
        mask = K.greater( y_true[0][0], 0)
        res = tf.math.add(loss , dist2)
        mask = tf.math.logical_not(mask)
        #return res
        if mask:
            return res
        else:
            return 0.0

    def __init__(self, ds_selection = ""):
        global model_loss
        device = torch.device('cpu')

        self.ds_selection = ds_selection
        itd = manipulating_images_better_v2.ImagesToData(ds_selection = self.ds_selection)
        itd.bf_ml()

        self.CXT = itd.CXT
        self.CYT = itd.CYT
        self.FXT = itd.FXT
        self.FYT = itd.FYT
        self.MXT = itd.MXT
        self.MYT = itd.MYT
        self.size = itd.size

        batch_size = 1
        self.batch_size = batch_size

        validation_split = 0.1
        
        self.chindatagen = ImageDataGenerator(
            validation_split=validation_split,
            #rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        self.chinvaldatagen = ImageDataGenerator(
            validation_split=validation_split,
            #rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        self.frendatagen = ImageDataGenerator(
            validation_split=validation_split,
            #rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        self.frenvaldatagen = ImageDataGenerator(
            validation_split=validation_split,
            #rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        ######################################################################################
        ############################# MODEL GENERATION #######################################
        input = Input((itd.size, itd.size, 3))
        x = EfficientNetB3( #tf.keras.applications.efficientnet.EfficientNetB0
            input_shape=(itd.size, itd.size, 3),
            weights='imagenet',
            include_top=False)(input)
        
        #model.summary()
        x = Flatten()(x)
        x = Dropout(0.15)(x)
        x = Dense(20, activation='relu', name='feature_extractor')(x)
        chin = Flatten()(x)
        chin = Dense(1, activation='sigmoid', name='dense')(chin)
        fren = Flatten()(x)
        fren = Dense(1, activation='sigmoid', name='dense_1')(fren)
        #chin.summary()
        #fren.summary()
        #inputs = Input(shape=(itd.size, itd.size, 3))
        model = Model(inputs=input,
                     outputs = [chin,fren],
                     name= 'model')
        #model.summary()
        model.trainable = True
        for layer in model.layers[1].layers:
            layer.trainable = False
        #model.summary()
        # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
        for layer in model.layers[1].layers[-4:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

        #model.summary()
        self.model = model

        ####################################################################################
        ###################### TRAINING LAST LAYERS AND FINE TUNING ########################
        print('RETRAINING')
        
        ep = 100
        verbose_param = 1
        
        lr_reduce = ReduceLROnPlateau(monitor='val_dense_accuracy', factor=0.2, patience=3, verbose=1, mode='max', min_lr=1e-8)
        #checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor= 'val_accuracy', mode= 'max', save_best_only = True, verbose= 0)
        early = EarlyStopping(monitor='val_dense_accuracy', min_delta=0.001, patience=13, verbose=1, mode='auto')

        lr_reduce_1 = ReduceLROnPlateau(monitor='val_dense_1_accuracy', factor=0.2, patience=3, verbose=1, mode='max', min_lr=1e-8)
        #checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor= 'val_accuracy', mode= 'max', save_best_only = True, verbose= 0)
        early_1 = EarlyStopping(monitor='val_dense_1_accuracy', min_delta=0.001, patience=13, verbose=1, mode='auto')
        
        learning_rate= 4e-3
        learning_rate_fine = 1e-8
        
        adam = optimizers.Adam(learning_rate)
        sgd = tf.keras.optimizers.SGD(learning_rate)
        rmsprop = tf.keras.optimizers.RMSprop(learning_rate)
        adadelta = tf.keras.optimizers.Adadelta(learning_rate)
        adagrad = tf.keras.optimizers.Adagrad(learning_rate)
        adamax = tf.keras.optimizers.Adamax(learning_rate)

        optimizer = sgd
        fine_optimizer = optimizers.SGD(learning_rate_fine)

        #model.compile(loss=[custom_loss_w1, custom_loss_w2], optimizer=optimizer, metrics=["accuracy"])
        
        
        if self.ds_selection == "chinese":
            print('chinese')
            dataset = self.chindatagen.flow_from_directory('../../' + working_directory ,
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            color_mode = 'rgb',
            class_mode = 'binary',
            classes = ['spente', 'accese'],
            subset = 'training',
            follow_links=True)

            dataset_val = self.chinvaldatagen.flow_from_directory('../../'+ working_directory ,
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            classes = ['spente', 'accese'],
            color_mode = 'rgb',
            subset = 'validation',
            follow_links=True)
            
            model.compile(loss=[self.custom_loss_w1 , self.custom_loss_w2], optimizer=optimizer, metrics=["accuracy"])
            
            history = self.model.fit(dataset,
            epochs=ep, validation_data=(dataset_val), 
            callbacks=[early_1, lr_reduce_1],verbose=verbose_param, batch_size=batch_size)
            

            
            
        if self.ds_selection == "french":
            print('french')
            chinese = self.chindatagen.flow_from_directory('../../' + working_directory,
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            color_mode = 'rgb',
            class_mode = 'binary',
            #classes = ['chinese', 'french', 'accese', 'spente'],
            subset = 'training',
            follow_links=True)

            chinese_val = self.chinvaldatagen.flow_from_directory('../../'+ working_directory,
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            #classes = ['chinese', 'french', 'accese', 'spente'],
            color_mode = 'rgb',
            subset = 'validation',
            follow_links=True)

            french = self.frendatagen.flow_from_directory('../../' + working_directory,
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'sparse',
            color_mode = 'rgb',
            subset = 'training',
            follow_links=True)

            french_val = self.frenvaldatagen.flow_from_directory('../../' + working_directory,
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'sparse',
            color_mode = "rgb",
            subset = 'validation',
            follow_links=True)

            history = self.model.fit(X , y,
            epochs=ep, validation_data=(X_val, y_val), 
            callbacks=[early_1, lr_reduce_1],verbose=verbose_param, batch_size=batch_size)
        
        self.M = model
            
            
        ##############################################################
        ############## PLOT SOME RESULTS ############################
        plot = False
        if plot:
            train_acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']
            No_Of_Epochs = range(ep)
            train_acc_x = range(len(train_acc))
            val_acc_x = range(len(train_acc))
            train_loss_x = range(len(train_acc))
            val_loss_x = range(len(train_acc))

            plt.plot(train_acc_x, train_acc, marker = 'o', color = 'blue', markersize = 10, 
                            linewidth = 1.5, label = 'Training Accuracy')
            plt.plot(val_acc_x, val_acc, marker = '.', color = 'red', markersize = 10, 
                            linewidth = 1.5, label = 'Validation Accuracy')

            plt.title('Training Accuracy and Testing Accuracy w.r.t Number of Epochs')

            plt.legend()

            plt.figure()

            plt.plot(train_loss_x, train_loss, marker = 'o', color = 'blue', markersize = 10, 
                            linewidth = 1.5, label = 'Training Loss')
            plt.plot(val_loss_x, val_acc, marker = '.', color = 'red', markersize = 10, 
                            linewidth = 1.5, label = 'Validation Loss')

            plt.title('Training Loss and Testing Loss w.r.t Number of Epochs')

            plt.legend()

            plt.show()

                

       
        