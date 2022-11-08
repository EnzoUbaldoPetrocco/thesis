#! /usr/bin/env python3
import tensorflow  as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from keras import layers, Model, optimizers
from keras.layers import Input, Dense, Flatten
import cv2
from skimage.color import rgb2gray
import pandas as pd
import math
import numpy as np
from tensorflow.keras.preprocessing import image

checkpoint_dir = ""
latest = ""

class RModel:

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



    def __init__(self, ds_selection, lamb):
        self.lamb = lamb
        self.ds_selection = ds_selection
        #itd = manipulating_images_better.ImagesToData()
        #self.itd = itd
        self.size = 128
        input = Input((self.size, self.size, 3))
        x = ResNet50( #tf.keras.applications.efficientnet.EfficientNetB0
            input_shape=(self.size, self.size, 3),
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

        learning_rate= 5e-4
        adam = optimizers.Adam(learning_rate)
        optimizer = adam
        
        if ds_selection == "chinese":
            self.model.compile(loss=[self.custom_loss_w2, self.custom_loss_w1], optimizer=optimizer, metrics=["accuracy"],  loss_weights=[1,1])
        if ds_selection == "french":
            self.model.compile(loss=[self.custom_loss_w2, self.custom_loss_w1], optimizer=optimizer, metrics=["accuracy"],  loss_weights=[1,1])
        
        #model.load_weights(latest)
        self.model.load_weights('./checkpoints/' + self.ds_selection +'/my_checkpoint' + str(self.lamb))

    
    def evaluate_sigmoid(self,y_pred, im_cult_info):
        if y_pred[im_cult_info][0][0]<0.5:
                return 0
        else:
                return 1

    def manage_size(self,im):
        dimensions = im.shape
        while dimensions[0]>self.size or dimensions[1]>self.size:
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
        list_size.append(math.floor((self.size - height)/2))
        list_size.append(math.ceil((self.size - height)/2))
        list_size.append(math.floor((self.size - width)/2))
        list_size.append(math.ceil((self.size - width)/2))
        return list_size

    def evaluate_image(self, im, im_cult_info=0):
        im = self.preprocessing(im)
        
        Y_pred = self.model.predict(im, verbose = 0)
        y = self.evaluate_sigmoid(Y_pred,im_cult_info)
        #print("Y_pred: Out0", str(Y_pred[0][0]), "Out1: ",str(Y_pred[1][0]) )
        #print("y: ", str(y))
        #print("y_true: ", str(y_true))
        
        return y

    def preprocessing(self, im):
        im = self.manage_size(im)
        dimensions = im.shape
        tblr = self.get_dimensions(dimensions[0],dimensions[1])
        im = cv2.copyMakeBorder(im, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])
        im = rgb2gray(im)
        im = pd.DataFrame(im).to_numpy()
        im = cv2.merge([im,im,im])*255
        im = image.img_to_array(im)
        im = np.expand_dims(im, axis=0)
        '''plt.figure()
        plt.imshow(im[0])
        plt.show()'''
        return im


