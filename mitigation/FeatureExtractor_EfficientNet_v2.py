#! /usr/bin/env python3
from audioop import rms
from re import I
from unicodedata import name
import manipulating_images_better
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
'''config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)'''
import torch
import os

working_directory = 'MITIGATION'
model_loss = 0
BATCH_SIZE = 1
lamb = .0


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
def custom_loss_w1(y_true,y_pred):
    w1 = K.get_value(model_loss.layers[len(model_loss.layers)-1].kernel)
    w2 = K.get_value(model_loss.layers[len(model_loss.layers)-1].kernel)

    weights1 = np.array([i[0] for i in w1])
    weights2 = np.array([i[1] for i in w2])
    
    dist = np.linalg.norm(weights1-weights2)
    dist2 = lamb*dist*dist
    dist2 = [dist2,dist2]
    dist2 = tf.convert_to_tensor(dist2)
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    return dist2 + loss



class FeatureExtractor:
    def __init__(self, ds_selection = ""):
        global model_loss
        device = torch.device('cpu')

        self.ds_selection = ds_selection
        itd = manipulating_images_better.ImagesToData(ds_selection = self.ds_selection)
        itd.bf_ml()

        self.CXT = itd.CXT
        self.CYT = itd.CYT
        self.FXT = itd.FXT
        self.FYT = itd.FYT
        self.MXT = itd.MXT
        self.MYT = itd.MYT
        self.size = itd.size

        batch_size = 1

        validation_split = 0.1
        
        chindatagen = ImageDataGenerator(
            validation_split=validation_split,
            #rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        chinvaldatagen = ImageDataGenerator(
            validation_split=validation_split,
            #rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        frendatagen = ImageDataGenerator(
            validation_split=validation_split,
            #rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        frenvaldatagen = ImageDataGenerator(
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
        model_loss = model

        ####################################################################################
        ###################### TRAINING LAST LAYERS AND FINE TUNING ########################
        print('RETRAINING')
        
        ep = 100
        eps_fine = 10
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

        model.compile(loss=custom_loss_w1, optimizer=optimizer, metrics=["accuracy"])
        
        
        if self.ds_selection == "chinese":
            print('chinese')
            chinese = chindatagen.flow_from_directory('../../' + working_directory,
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            color_mode = 'rgb',
            class_mode = 'binary',
            #classes = ['chinese', 'french', 'accese', 'spente'],
            subset = 'training',
            follow_links=True)

            chinese_val = chinvaldatagen.flow_from_directory('../../'+ working_directory,
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            #classes = ['chinese', 'french', 'accese', 'spente'],
            color_mode = 'rgb',
            subset = 'validation',
            follow_links=True)

            '''for i in chinese:
                print(i)
                plt.figure()
                plt.imshow(i[0][0]*1/255)
                plt.show()'''

            Number_Of_Training_Images = chinese.classes.shape[0]
            steps_per_epoch = Number_Of_Training_Images/batch_size

            history = model.fit(chinese,
            epochs=ep, validation_data=chinese_val, 
            callbacks=[early, lr_reduce, early_1, lr_reduce_1],verbose=verbose_param)
            
        if self.ds_selection == "french":
            print('french')
            french = frendatagen.flow_from_directory('../../' + working_directory,
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'sparse',
            color_mode = 'rgb',
            subset = 'training',
            follow_links=True)

            french_val = frenvaldatagen.flow_from_directory('../../' + working_directory,
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'sparse',
            color_mode = "rgb",
            subset = 'validation',
            follow_links=True)

            Number_Of_Training_Images = french.classes.shape[0]
            steps_per_epoch = Number_Of_Training_Images/batch_size

            history = model.fit(french, epochs=ep, 
            validation_data=french_val, callbacks=[early, lr_reduce], verbose=verbose_param)
        
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

                

       
        