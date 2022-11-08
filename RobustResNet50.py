#! /usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from keras import Model, layers, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Flatten, Input
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

import manipulating_images_better

'''config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)'''
import pandas as pd
import torch
import random

working_directory = 'MITIGATION'
im_path = '../th/'
weight1 = 0
weight2 = 0
BATCH_SIZE = 1

def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)

    '''Add random noise to an image'''
    VARIABILITY = 60
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)

    return image
  
    
def dummy_loss(y_true, y_pred):
    return 0.0


class FeatureExtractor:


    def custom_loss_w1(self,y_true,y_pred):
        # Calculate lambda * ||Wc - Wf||^2
        weights1 = self.model.layers[len(self.model.layers)-1].kernel
        weights2 = self.model.layers[len(self.model.layers)-2].kernel
        dist = tf.norm(weights1-weights2, ord='euclidean')
        dist2 = tf.multiply(tf.multiply(dist,dist) , self.lamb)
        # Loss
        loss = tf.keras.losses.binary_crossentropy(y_true[0][1], y_pred[0])
        mask = tf.math.multiply(0.5, tf.math.add((tf.math.add(y_true[0][0], 0.0)), tf.math.abs(tf.math.subtract(y_true[0][0], 0.0))))     
        res = tf.math.add(loss , dist2)
        if mask > 0 :
            return res
        else:
            return 0.0

    #@tf.function
    def custom_loss_w2(self, y_true,y_pred):
        # Calculate lambda * ||Wc - Wf||^2
        weights1 = self.model.layers[len(self.model.layers)-1].kernel
        weights2 = self.model.layers[len(self.model.layers)-2].kernel
        dist = tf.norm(weights1-weights2, ord='euclidean')
        dist2 = tf.multiply(tf.multiply(dist,dist) , self.lamb)
        # Loss
        loss = tf.keras.losses.binary_crossentropy(y_true[0][1], y_pred[0])
        mask = tf.math.multiply(0.5, tf.math.add((tf.math.add(y_true[0][0], 0)),tf.math.abs(tf.math.subtract(y_true[0][0], 0))))  ##.5*( a + b + |a - b |)
        res = tf.math.add(loss , dist2)
        #return res
        if mask > 0 :
            return 0.0
        else:
            return res

    def create_dataframe(self, rootDir, e, on):
        df = []
        '''for dirName, subdirList, fileList in os.walk(rootDir):
            for fname in fileList:
                df = pd.read_csv(fname)'''
        paths = [path.parts[-3:] for path in Path(rootDir).rglob('*.jpeg')]
        #ds = []
        im_list = []
        #y_list = []
        e_list = []
        on_list = []
        for i in paths:
            i = list(i)
            im = cv2.imread(rootDir + '/' + str(i[2]))
            im = tf.convert_to_tensor(im)
            im_list.append(im)
            e_list.append(e)
            on_list.append(on)
        df = pd.DataFrame(data={ 'input_1': im_list, 'e': e_list, 'on': on_list})
        return df

    def dataset_management(self):
        dir = im_path + working_directory + '/french/accese'
        french_on = self.create_dataframe(dir, 0.0 ,1.0)
        dir = im_path + working_directory + '/french/spente'
        french_off = self.create_dataframe(dir,0.0 ,0.0)
        dir = im_path + working_directory + '/chinese/accese'
        chinese_on = self.create_dataframe(dir, 1.0 ,1.0)
        dir = im_path + working_directory + '/chinese/spente'
        chinese_off = self.create_dataframe(dir,1.0,0.0)
        french = pd.concat([french_off, french_on], ignore_index=True)
        chinese = pd.concat([chinese_off, chinese_on], ignore_index=True)
        ds = pd.concat([chinese, french], ignore_index=True)
        ds = shuffle(ds)
        return ds
        
    def __init__(self, ds_selection = "", lamb = 0):
        global model_loss
        device = torch.device('cpu')

        self.ds_selection = ds_selection
        itd = manipulating_images_better.ImagesToData(ds_selection = self.ds_selection)
        itd.bf_ml()

        self.lamb = lamb

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
        rotation_range = 25
        bright_rng = [0.1, 1.5]
        
        self.chindatagen = ImageDataGenerator(
            validation_split=validation_split,
            brightness_range=bright_rng,
            rotation_range=rotation_range,
            horizontal_flip=True,
    preprocessing_function=to_grayscale_then_rgb)

        self.chinvaldatagen = ImageDataGenerator(
            validation_split=validation_split,
            brightness_range=bright_rng,
            rotation_range=rotation_range,
            horizontal_flip=True,
    preprocessing_function=to_grayscale_then_rgb)

        self.frendatagen = ImageDataGenerator(
            validation_split=validation_split,
            brightness_range=bright_rng,
            rotation_range=rotation_range,
            horizontal_flip=True,
    preprocessing_function=to_grayscale_then_rgb)

        self.frenvaldatagen = ImageDataGenerator(
            validation_split=validation_split,
            brightness_range=bright_rng,
            rotation_range=rotation_range,
            horizontal_flip=True,
    preprocessing_function=to_grayscale_then_rgb)

        ######################################################################################
        ############################# MODEL GENERATION #######################################
        
        input = Input((itd.size, itd.size, 3))

        x = ResNet50( 
            input_shape=(itd.size, itd.size, 3),
            weights='imagenet',
            include_top=False)(input)
        
        #model.summary()
        x = Flatten()(x)
        chin = Dense(1, activation='sigmoid', name='dense')(x)
        fren = Dense(1, activation='sigmoid', name='dense_1')(x)
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

        #model.summary()
        self.model = model

        ####################################################################################
        ###################### TRAINING LAST LAYERS AND FINE TUNING ########################
        print('RETRAINING')
        
        ep = 100
        verbose_param = 1
        #self.batch_end = self.CustomCallback(self.model, self.lamb)
        
        lr_reduce = ReduceLROnPlateau(monitor='val_dense_accuracy', factor=0.2, patience=2, verbose=1, mode='max', min_lr=1e-8)
        #checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor= 'val_accuracy', mode= 'max', save_best_only = True, verbose= 0)
        early = EarlyStopping(monitor='val_dense_accuracy', min_delta=0.001, patience=8, verbose=1, mode='auto')

        lr_reduce_1 = ReduceLROnPlateau(monitor='val_dense_1_accuracy', factor=0.2, patience=2, verbose=1, mode='max', min_lr=1e-8)
        #checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor= 'val_accuracy', mode= 'max', save_best_only = True, verbose= 0)
        early_1 = EarlyStopping(monitor='val_dense_1_accuracy', min_delta=0.001, patience=8, verbose=1, mode='auto')
        
        learning_rate= 1e-4
        learning_rate_fine = 1e-8
        
        adam = optimizers.Adam(learning_rate)
        sgd = tf.keras.optimizers.SGD(learning_rate)
        rmsprop = tf.keras.optimizers.RMSprop(learning_rate)
        adadelta = tf.keras.optimizers.Adadelta(learning_rate)
        adagrad = tf.keras.optimizers.Adagrad(learning_rate)
        adamax = tf.keras.optimizers.Adamax(learning_rate)

        optimizer = adam
        fine_optimizer = optimizers.SGD(learning_rate_fine)

        
        if self.ds_selection == "chinese":
            print('chinese')
            ds = self.dataset_management()
            train_size = int(0.9*ds.shape[0])
            val_size = int(0.1*ds.shape[0])
            dataset = ds.head(train_size)
            dataset_val = ds.tail(val_size)
            self.model.compile(loss=[self.custom_loss_w1, self.custom_loss_w2], optimizer=optimizer, metrics=["accuracy"],  loss_weights=[1,1])#, run_eagerly=True)
            dataset = dataset.to_numpy()
            dataset_val = dataset_val.to_numpy()
            
            X = []
            y = []
            X_val = []
            y_val = []
            for i in dataset:
                '''
                plt.figure()
                plt.imshow(i[0])
                plt.show()

                '''
                
                X.append(i[0])
                y_temp = [i[1], i[2]]
                y.append(y_temp)

            for i in dataset_val:
                X_val.append(i[0])
                y_temp = [i[1], i[2]]
                y_val.append(y_temp)
            X = tf.stack(X)
            y = tf.stack(y)
            X_val = tf.stack(X_val)
            y_val = tf.stack(y_val)
            
            tf.get_logger().setLevel('ERROR')
            print(np.linalg.norm(np.array([i[0] for i in self.model.layers[len(self.model.layers)-2].get_weights()])-np.array([i[0] for i in self.model.layers[len(self.model.layers)-1].get_weights()])))
            history = self.model.fit(X , y,
            epochs=ep, validation_data=(X_val, y_val), 
            callbacks=[early, lr_reduce],verbose=verbose_param, batch_size=batch_size)
            print(np.linalg.norm(np.array([i[0] for i in self.model.layers[len(self.model.layers)-2].get_weights()])-np.array([i[0] for i in self.model.layers[len(self.model.layers)-1].get_weights()])))
            
        if self.ds_selection == "french":
            print('french')
            ds = self.dataset_management()
            train_size = int(0.9*ds.shape[0])
            val_size = int(0.1*ds.shape[0])
            dataset = ds.head(train_size)
            dataset_val = ds.tail(val_size)
            self.model.compile(loss=[self.custom_loss_w2, self.custom_loss_w1], optimizer=optimizer, metrics=["accuracy"],  loss_weights=[1,1])#, run_eagerly=True)
            dataset = dataset.to_numpy()
            dataset_val = dataset_val.to_numpy()
            
            X = []
            y = []
            X_val = []
            y_val = []
            for i in dataset:
                X.append(i[0])
                y_temp = [i[1], i[2]]
                y.append(y_temp)

            for i in dataset_val:
                X_val.append(i[0])
                y_temp = [i[1], i[2]]
                y_val.append(y_temp)
            X = tf.stack(X)
            y = tf.stack(y)
            X_val = tf.stack(X_val)
            y_val = tf.stack(y_val)

            tf.get_logger().setLevel('ERROR')
            print(np.linalg.norm(np.array([i[0] for i in self.model.layers[len(self.model.layers)-2].get_weights()])-np.array([i[0] for i in self.model.layers[len(self.model.layers)-1].get_weights()])))
            history = self.model.fit(X , y,
            epochs=ep, validation_data=(X_val, y_val), 
            callbacks=[early_1, lr_reduce_1],verbose=verbose_param, batch_size=batch_size)
            print(np.linalg.norm(np.array([i[0] for i in self.model.layers[len(self.model.layers)-2].get_weights()])-np.array([i[0] for i in self.model.layers[len(self.model.layers)-1].get_weights()])))
            

        self.M = self.model

        model.save_weights('./checkpoints/' + self.ds_selection +'/my_checkpoint' + str(self.lamb))
         
            
            
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

                

       
        