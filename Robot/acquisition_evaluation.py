#! /usr/bin/env python3
import os
import tensorflow  as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from keras import layers, models, Model, optimizers
from keras.layers import Input, Lambda, Dense, Flatten,Dropout, MaxPooling3D
import manipulating_images_better
import cv2
from skimage.color import rgb2gray
import pandas as pd
import math

checkpoint_dir = ""
latest = ""

class RModel:

    def __init__(self, ds_selection, lamb):
        self.lamb = lamb
        itd = manipulating_images_better.ImagesToData()
        self.itd = itd
        input = Input((itd.size, itd.size, 3))
        x = ResNet50( #tf.keras.applications.efficientnet.EfficientNetB0
            input_shape=(itd.size, itd.size, 3),
            weights='imagenet',
            include_top=False)(input)

        #model.summary()
        x = Flatten()(x)
        chin = Dense(1, activation='sigmoid', name='dense')(x)
        fren = Dense(1, activation='sigmoid', name='dense_1')(x)
        self.model = Model(inputs=input,
                        outputs = [chin,fren],
                        name= 'model')
        self.model.trainable = True
        for layer in self.model.layers[1].layers:
            layer.trainable = False
        for layer in self.model.layers[1].layers[-1:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

        learning_rate= 4e-4
        adam = optimizers.Adam(learning_rate)
        optimizer = adam
        if ds_selection == "chinese":
            self.model.compile(loss=[self.custom_loss_w2, self.custom_loss_w1], optimizer=optimizer, metrics=["accuracy"],  loss_weights=[1,1])
        if ds_selection == "french":
            self.model.compile(loss=[self.custom_loss_w2, self.custom_loss_w1], optimizer=optimizer, metrics=["accuracy"],  loss_weights=[1,1])
        
        #model.load_weights(latest)
        self.model.load_weights('./checkpoints/' + self.ds_selection +'/my_checkpoint' + str(self.lamb))
    
    def evaluate_sigmoid(self,y_pred):
        if y_pred<0.5:
                return 0
        else:
                return 1

    def manage_size(self,im):
        dimensions = im.shape
        while dimensions[0]>self.itd.size or dimensions[1]>self.itd.size:
            width = int(im.shape[1] * 0.9)
            height = int(im.shape[0] * 0.9)
            dim = (width, height)
            try:
                im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA )
            except:
                print('Error: resize did not worked')
            dimensions = im.shape
        return im

    def get_dimensions(self,height, width):
        list_size = []
        list_size.append(math.floor((self.itd.size - height)/2))
        list_size.append(math.ceil((self.itd.size - height)/2))
        list_size.append(math.floor((self.itd.size - width)/2))
        list_size.append(math.ceil((self.itd.size - width)/2))
        return list_size

    def evaluate_image(self, im):
        im = self.preprocessing(im)
        Y_pred = self.model.predict(im)
        y = self.evaluate_sigmoid(y_pred=Y_pred)
        return y

    def preprocessing(self, im):
        im = self.manage_size(im)
        dimensions = im.shape
        tblr = self.get_dimensions(dimensions[0],dimensions[1])
        im = cv2.copyMakeBorder(im, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])
        im = rgb2gray(im)
        im_obj = pd.DataFrame(im).to_numpy()
        return im_obj


