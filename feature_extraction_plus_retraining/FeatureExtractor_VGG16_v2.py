#! /usr/bin/env python3
from audioop import rms
from unicodedata import name
import manipulating_images_better
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
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
from keras.layers import Input, Lambda, Dense, Flatten,Dropout
from keras.models import Sequential


BATCH_SIZE = 1

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

        batch_size = 8
        batch_fit = 8

        validation_split = 0.3

        chindatagen = ImageDataGenerator(
            validation_split=validation_split,
            rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        chinvaldatagen = ImageDataGenerator(
            validation_split=validation_split,
            rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        frendatagen = ImageDataGenerator(
            validation_split=validation_split,
            rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        frenvaldatagen = ImageDataGenerator(
            validation_split=validation_split,
            rescale=1/255,
    preprocessing_function=to_grayscale_then_rgb)

        ######################################################################################
        ############################# MODEL GENERATION #######################################
        #model = VGG16(weights='imagenet', include_top=False,  input_shape=(itd.size,itd.size,3))
        model_pre = VGG19(weights='imagenet', include_top=False,  input_shape=(itd.size,itd.size,3))
        # Create the model
        model = Sequential()
        # Add the vgg convolutional base model
        model.add(model_pre)
        model.trainable = False
        model.summary()
        # Add new layers
        model.add(Flatten())
        model.add(Dense(300, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(150, activation='relu', name = 'feature_extractor'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        # Show a summary of the model. Check the number of trainable parameters
        # Freeze four convolution blocks
        model.trainable = True
        for layer in model.layers[:len(model.layers)-5]:
            layer.trainable = False
        model.summary()

        ####################################################################################
        ###################### RETRAINING ###############################################
        print('RETRAINING')
        
        ep = 30
        
        lr_reduce = ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=4, verbose=1, mode='max', min_lr=1e-7)
        #checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor= 'val_accuracy', mode= 'max', save_best_only = True, verbose= 0)
        early = EarlyStopping(monitor='accuracy', min_delta=0.001, patience=16, verbose=1, mode='auto')
        
        learning_rate= 1e-4
        '''lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.9)
        sgd = tf.keras.optimizers.SGD(learning_rate=lr_schedule)'''
        adam = optimizers.Adam(learning_rate)
        sgd = tf.keras.optimizers.SGD(learning_rate)
        rmsprop = tf.keras.optimizers.RMSprop(learning_rate)
        adadelta = tf.keras.optimizers.Adadelta(learning_rate)
        adagrad = tf.keras.optimizers.Adagrad(learning_rate)
        adamax = tf.keras.optimizers.Adamax(learning_rate)

        model.compile(loss="binary_crossentropy", optimizer=rmsprop, metrics=["accuracy"])
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

            history = model.fit(chinese, 
            #batch_size = batch_size, 
            epochs=ep, validation_data=chinese_val, 
            #steps_per_epoch = steps_per_epoch,
            callbacks=[early, lr_reduce],verbose=1)
            
        if self.ds_selection == "french":
            print('french')
            french = frendatagen.flow_from_directory('../../FE/' + ds_selection + '/french',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = 'grayscale',
            subset = 'training')

            french_val = frenvaldatagen.flow_from_directory('../../FE/' + ds_selection + '/french',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = "grayscale",
            subset = 'validation')

            Number_Of_Training_Images = french.classes.shape[0]
            steps_per_epoch = Number_Of_Training_Images/batch_size

            history = model.fit(french,  
            #steps_per_epoch = steps_per_epoch,
            #batch_size = batch_size, 
            epochs=ep, validation_data=french_val, callbacks=[early, lr_reduce], verbose=0)
            
        if self.ds_selection == "mix":
            print('mix')
            chinese = chindatagen.flow_from_directory('../../FE/' + ds_selection + '/chinese',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = "grayscale",
            subset = 'training')

            chinese_val = chinvaldatagen.flow_from_directory('../../FE/' + ds_selection + '/chinese',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = "grayscale",
            subset = 'validation')

            french = frendatagen.flow_from_directory('../../FE/' + ds_selection + '/french',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = "grayscale",
            subset = 'training')

            french_val = frenvaldatagen.flow_from_directory('../../FE/' + ds_selection + '/french',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = "grayscale",
            subset = 'validation')

            dataset = tf.data.Dataset.zip((chinese, french))
            dataset_val = tf.data.Dataset.zip((chinese_val, french_val))

            Number_Of_Training_Images = dataset.classes.shape[0]
            steps_per_epoch = Number_Of_Training_Images/batch_size

            history = model.fit(dataset,  
            #steps_per_epoch = steps_per_epoch,
            #batch_size = batch_size, 
            epochs=ep, validation_data=dataset_val, callbacks=[early, lr_reduce], verbose=1)
            
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
                            linewidth = 2, label = 'Training Accuracy')
            plt.plot(val_acc_x, val_acc, marker = '.', color = 'red', markersize = 10, 
                            linewidth = 2, label = 'Validation Accuracy')

            plt.title('Training Accuracy and Testing Accuracy w.r.t Number of Epochs')

            plt.legend()

            plt.figure()

            plt.plot(train_loss_x, train_loss, marker = 'o', color = 'blue', markersize = 10, 
                            linewidth = 2, label = 'Training Loss')
            plt.plot(val_loss_x, val_acc, marker = '.', color = 'red', markersize = 10, 
                            linewidth = 2, label = 'Validation Loss')

            plt.title('Training Loss and Testing Loss w.r.t Number of Epochs')

            plt.legend()

            plt.show()
                

        #################################################
        ############# FEATURE EXTRACTION ################
        #print(model.layers[-2])
        model = Model(inputs=model.inputs, outputs=model.get_layer(name="feature_extractor").output)
        #model.layers.pop()
        #model.layers.pop()
        model.summary()
        
        print('FEATURE EXTRACTION')
        features = []
        for i in CX:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0])
            
        self.CX = np.array(features)
        self.CY = CY
        print(np.shape(CX))
        print(np.shape(CX[0]))
        print(np.shape(CX[0][0]))
        print(CX[0])

        features = []
        for i in CXT:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0])
        
        self.CXT = np.array(features)
        self.CYT = CYT

        features = []
        for i in FX:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0])
        
        self.FX = np.array(features)
        self.FY = FY

        features = []
        for i in FXT:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0])
        
        self.FXT = np.array(features)
        self.FYT = FYT

        features = []
        for i in MX:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0])
        
        self.MX = np.array(features)
        self.MY = MY

        features = []
        for i in MXT:
            x = np.reshape(i, (itd.size,itd.size))
            x = cv2.merge([x,x,x])
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            feature = model.predict(x, verbose = 0)
            features.append(feature[0])
        
        self.MXT = np.array(features)
        self.MYT = MYT

        

        








