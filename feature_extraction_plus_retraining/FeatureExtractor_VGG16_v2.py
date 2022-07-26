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

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
        else:
            print('no gpus')

        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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

        batch_size = 1

        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2)
        chinese = datagen.flow_from_directory('../../FE/chinese',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            subset = 'training')

        chinese_val = datagen.flow_from_directory('../../FE/chinese',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            subset = 'validation')

        french = datagen.flow_from_directory('../../FE/french',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            subset = 'training')

        french_val = datagen.flow_from_directory('../../FE/french',
            target_size = (itd.size, itd.size),
            batch_size = batch_size,
            class_mode = 'binary',
            subset = 'validation')

        
        
        #model = VGG16(weights='imagenet', include_top=False,  input_shape=(itd.size,itd.size,3))
        model = ResNet50(weights='imagenet', include_top=False,  input_shape=(itd.size,itd.size,3))
        ####################################################################################
        ######################## RETRAINING  ###############################################
        print('RETRAINING')
        
        ep = 20
        # Freeze four convolution blocks
        for layer in model.layers[:len(model.layers)-1]:
            layer.trainable = False


        lr_reduce = ReduceLROnPlateau(monitor='binary_accuracy', factor=0.6, patience=6, verbose=1, mode='max', min_lr=5e-5)
        #checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor= 'val_accuracy', mode= 'max', save_best_only = True, verbose= 0)
        early = EarlyStopping(monitor='binary_accuracy', min_delta=0.001, patience=9, verbose=1, mode='auto')
        
        learning_rate= 1e-5

        model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(learning_rate), metrics=["binary_accuracy"])
        #history = model.fit(X_train, y_train, batch_size = 1, epochs=50, validation_data=(X_test,y_test), callbacks=[lr_reduce,checkpoint])

        X_train = []
        X_test = []
        y_train = []
        y_test = []

        prop = 1/22

        prop_dsts = 0.8

        if self.ds_selection == "chinese":
            print('chinese')
            


            history = model.fit(chinese, batch_size = batch_size, epochs=ep, validation_data=chinese_val, callbacks=[early, lr_reduce],steps_per_epoch= int(len(CX)), verbose=0)
            
        if self.ds_selection == "french":
            print('french')

            history = model.fit(french, batch_size = batch_size, epochs=ep, validation_data=french_val, callbacks=[early, lr_reduce], verbose=1)
            
        if self.ds_selection == "mix":
            print('mix')

            dataset = tf.data.Dataset.zip((chinese, french))
            dataset_val = tf.data.Dataset.zip((chinese_val, french_val))

            history = model.fit(dataset, batch_size = batch_size, epochs=ep, validation_data=dataset_val, callbacks=[early, lr_reduce], verbose=1)
            

        

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

        

        








