#! /usr/bin/env python3
from curses.panel import new_panel
from tabnanny import verbose
import manipulating_images_better
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, Model, optimizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import tensorflow as tf
from skimage.color import gray2rgb

BATCH_SIZE = 1

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


class FeatureExtractor:
    def __init__(self, ds_selection = ""):
        self.ds_selection = ds_selection
        itd = manipulating_images_better.ImagesToData()
        itd.bf_ml()

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
        
        #model = VGG16(weights='imagenet', include_top=False,  input_shape=(itd.size,itd.size,3))
        model = ResNet50(weights='imagenet', include_top=False,  input_shape=(itd.size,itd.size,3))
        ####################################################################################
        ######################## RETRAINING  ###############################################
        print('RETRAINING')
        batch_size = 1
        ep = 50
        # Freeze four convolution blocks
        for layer in model.layers[:len(model.layers)]:
            layer.trainable = False

        lr_reduce = ReduceLROnPlateau(monitor='binary_accuracy', factor=0.6, patience=6, verbose=1, mode='max', min_lr=5e-5)
        #checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor= 'val_accuracy', mode= 'max', save_best_only = True, verbose= 0)
        early = EarlyStopping(monitor='binary_accuracy', min_delta=0.001, patience=9, verbose=1, mode='auto')
        
        learning_rate= 5e-4

        model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=["binary_accuracy"])
        #history = model.fit(X_train, y_train, batch_size = 1, epochs=50, validation_data=(X_test,y_test), callbacks=[lr_reduce,checkpoint])

        X_train = []
        X_test = []
        y_train = []
        y_test = []

        prop = 1/3

        if self.ds_selection == "chinese":
            
            for (x, y) in zip(CX[0:int(len(CX)*(prop)*(3/4))],  
            CY[0:int(len(CY)*(prop)*(3/4))]):
                
                x = np.reshape(x, (itd.size,itd.size))
                x = gray2rgb(x)

                X_train.append(x)
                y_train.append(y)

            for (xt, yt) in zip(CX[int(len(CX)*(prop)*(3/4)):int(len(CX)*(prop))], 
            CY[int(len(CY)*(prop)*(3/4)):int(len(CY)*(prop))]):

                xt = np.reshape(xt, (itd.size,itd.size))
                xt = gray2rgb(xt)
                X_test.append(xt)
                y_test.append(yt)

            y_train = np.array(y_train)
            y_test = np.array(y_test)
            X_train = np.array(X_train)
            X_test = np.array(X_test)


            X_train = batch_generator(X_train, y_train, batch_size= batch_size)
            X_test =  batch_generator(X_test, y_test, batch_size= batch_size)


            history = model.fit(X_train, batch_size = batch_size, epochs=ep, validation_data=X_test, callbacks=[early], verbose=0)
            
        if self.ds_selection == "french":

            for (x, y) in zip(FX[0:int(len(FX)*(prop)*(3/4))],  
            FY[0:int(len(FY)*(prop)*(3/4))]):
                
                x = np.reshape(x, (itd.size,itd.size))
                x = gray2rgb(x)

                X_train.append(x)
                y_train.append(y)

            for (xt, yt) in zip(FX[int(len(FX)*(prop)*(3/4)):int(len(FX)*(prop))], 
            FY[int(len(FY)*(prop)*(3/4)):int(len(FY)*(prop))]):

                xt = np.reshape(xt, (itd.size,itd.size))
                xt = gray2rgb(xt)
                X_test.append(xt)
                y_test.append(yt)

            y_train = np.array(y_train)
            y_test = np.array(y_test)
            X_train = np.array(X_train)
            X_test = np.array(X_test)

            X_train = batch_generator(X_train, y_train, batch_size= batch_size)
            X_test =  batch_generator(X_test, y_test, batch_size= batch_size)


            history = model.fit(X_train, batch_size = batch_size, epochs=ep, validation_data=X_test, callbacks=[early], verbose=0)
            
        if self.ds_selection == "mix":

            for (x, y) in zip(MX[0:int(len(MX)*(prop)*(3/4))],  
            MY[0:int(len(MY)*(prop)*(3/4))]):
                
                x = np.reshape(x, (itd.size,itd.size))
                x = gray2rgb(x)

                X_train.append(x)
                y_train.append(y)

            for (xt, yt) in zip(MX[int(len(MX)*(prop)*(3/4)):int(len(MX)*(prop))], 
            MY[int(len(MY)*(prop)*(3/4)):int(len(MY)*(prop))]):

                xt = np.reshape(xt, (itd.size,itd.size))
                xt = gray2rgb(xt)
                X_test.append(xt)
                y_test.append(yt)

            y_train = np.array(y_train)
            y_test = np.array(y_test)
            X_train = np.array(X_train)
            X_test = np.array(X_test)

            X_train = batch_generator(X_train, y_train, batch_size= batch_size)
            X_test =  batch_generator(X_test, y_test, batch_size= batch_size)


            history = model.fit(X_train, batch_size = batch_size, epochs=ep, validation_data=X_test, callbacks=[early], verbose=0)
            
        CX = CX[int(len(CX)*(prop)): len(CX)-1]
        CY = CY[int(len(CY)*(prop)): len(CY)-1]
        FX = FX[int(len(FX)*(prop)): len(FX)-1]
        FY = FY[int(len(FY)*(prop)): len(FY)-1]
        MX = MX[int(len(MX)*(prop)): len(MX)-1]
        MY = MY[int(len(MY)*(prop)): len(MY)-1]

        

        #################################################
        ############# FEATURE EXTRACTION ################
        print('FEATURE EXTRACTION')
        features = []
        for i in CX:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0][0][0])
            
        self.CX = np.array(features)
        self.CY = CY

        features = []
        for i in CXT:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0][0][0])
        
        self.CXT = np.array(features)
        self.CYT = CYT

        features = []
        for i in FX:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0][0][0])
        
        self.FX = np.array(features)
        self.FY = FY

        features = []
        for i in FXT:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0][0][0])
        
        self.FXT = np.array(features)
        self.FYT = FYT

        features = []
        for i in MX:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0][0][0])
        
        self.MX = np.array(features)
        self.MY = MY

        features = []
        for i in MXT:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0][0][0])
        
        self.MXT = np.array(features)
        self.MYT = MYT

        

        








