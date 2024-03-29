#! /usr/bin/env python3

import manipulating_images_better
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from keras.layers import  Dense, Flatten,Dropout
from tensorflow.keras.applications.efficientnet import EfficientNetB3
'''config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)'''
import torch


BATCH_SIZE = 1

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


class FeatureExtractor:
    def __init__(self, ds_selection = ""):
        device = torch.device('cpu')

        self.ds_selection = ds_selection
        itd = manipulating_images_better.ImagesToData(ds_selection = self.ds_selection)
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
        
        model = tf.keras.Sequential([
        EfficientNetB3( #tf.keras.applications.efficientnet.EfficientNetB0
            input_shape=(itd.size, itd.size, 3),
            weights='imagenet',
            include_top=False
        )#,
        #L.GlobalAveragePooling2D()#,
        #L.Dense(1, activation='sigmoid')
    ])
        
        model.add(Flatten())
        model.add(Dropout(0.15))
        model.add(Dense(20, activation='relu', name='feature_extractor'))
        model.add(Dense(1, activation='sigmoid'))
        model.trainable = True
        '''for layer in model.layers[:len(model.layers)-4]:
            layer.trainable = False'''
        '''for layer in model.layers[0].layers[:len(model.layers[0].layers)-5]:
            layer.trainable = False'''
        for layer in model.layers[0].layers:
            layer.trainable = False
        # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
        for layer in model.layers[0].layers[-4:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        #model.summary()

        ####################################################################################
        ###################### TRAINING LAST LAYERS AND FINE TUNING ########################
        print('RETRAINING')
        
        ep = 100
        eps_fine = 10
        verbose_param = 0
        
        lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=3, verbose=1, mode='max', min_lr=1e-8)
        #checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor= 'val_accuracy', mode= 'max', save_best_only = True, verbose= 0)
        early = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=13, verbose=1, mode='auto')
        
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

        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        #history = model.fit(X_train, y_train, batch_size = 1, epochs=50, validation_data=(X_test,y_test), callbacks=[lr_reduce,checkpoint])
        
        if self.ds_selection == "chinese":
            print('chinese')
            chinese = chindatagen.flow_from_directory('../../FE/' + ds_selection + '/chinese',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            color_mode = 'rgb',
            class_mode = 'binary',
            subset = 'training')

            chinese_val = chinvaldatagen.flow_from_directory('../../FE/' + ds_selection + '/chinese',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = 'rgb',
            subset = 'validation')

            '''for i in chinese:
                plt.figure()
                plt.imshow(i[0][0])
                plt.show()'''

            

            Number_Of_Training_Images = chinese.classes.shape[0]
            steps_per_epoch = Number_Of_Training_Images/batch_size

            #os.environ['CUDA_VISIBLE_DEVICES'] = '0'

            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #print('Using device:' , device)
            #model

            history = model.fit(chinese, 
            #batch_size = batch_size, 
            epochs=ep, validation_data=chinese_val, 
            #steps_per_epoch = steps_per_epoch,
            callbacks=[early, lr_reduce],verbose=verbose_param)
            #device = torch.device('cpu')
            
        if self.ds_selection == "french":
            print('french')
            french = frendatagen.flow_from_directory('../../FE/' + ds_selection + '/french',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = 'rgb',
            subset = 'training')

            french_val = frenvaldatagen.flow_from_directory('../../FE/' + ds_selection + '/french',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = "rgb",
            subset = 'validation')

            Number_Of_Training_Images = french.classes.shape[0]
            steps_per_epoch = Number_Of_Training_Images/batch_size

            #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #print('Using device:' , device)
            history = model.fit(french, epochs=ep, 
            validation_data=french_val, callbacks=[early, lr_reduce], verbose=verbose_param)
            #device = torch.device('cpu')            
            
            
        if self.ds_selection == "mix":
            print('mix')
            dataset = chindatagen.flow_from_directory('../../FE/' + ds_selection,
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = "rgb",
            subset = 'training')

            dataset_val = chinvaldatagen.flow_from_directory('../../FE/' + ds_selection,
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = "rgb",
            subset = 'validation')

            
            Number_Of_Training_Images = dataset.classes.shape[0]
            steps_per_epoch = Number_Of_Training_Images/batch_size
            #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #print('Using device:' , device)
            history = model.fit(dataset,  
            steps_per_epoch = steps_per_epoch,
            epochs=ep, validation_data=dataset_val, callbacks=[early, lr_reduce], verbose=verbose_param)
            #device = torch.device('cpu')

            
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

                

        #################################################
        ############# FEATURE EXTRACTION ################
        #print(model.layers[-2])
        #model = Model(inputs=model.inputs, outputs=model.layers[:-2])
        model.layers.pop()        
        #model.summary()
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print('Using device:' , device)
        
        print('FEATURE EXTRACTION')
        features = []
        for i in CX:
            x = np.reshape(i, (itd.size,itd.size))*255
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0].flatten())

            
        self.CX = np.array(features)
        self.CY = CY

        features = []
        for i in CXT:
            x = np.reshape(i, (itd.size,itd.size))*255
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0].flatten())
        
        self.CXT = np.array(features)
        self.CYT = CYT

        features = []
        for i in FX:
            x = np.reshape(i, (itd.size,itd.size))*255
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0].flatten())
        
        self.FX = np.array(features)
        self.FY = FY

        features = []
        for i in FXT:
            x = np.reshape(i, (itd.size,itd.size))*255
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0].flatten())
        
        self.FXT = np.array(features)
        self.FYT = FYT

        features = []
        for i in MX:
            x = np.reshape(i, (itd.size,itd.size))*255
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0].flatten())
        
        self.MX = np.array(features)
        self.MY = MY

        features = []
        for i in MXT:
            x = np.reshape(i, (itd.size,itd.size))*255
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0].flatten())
        
        self.MXT = np.array(features)
        self.MYT = MYT

        
        #device = torch.device('cpu') 

        