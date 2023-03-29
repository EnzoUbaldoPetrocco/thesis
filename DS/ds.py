#! /usr/bin/env python3

import pathlib
import numpy as np
import skimage.color
import cv2
import matplotlib.pyplot as plt
import math
import pandas as pd
import random
import time
import os, shutil
from PIL import Image

n_ims = 1000

class DS:
  # utils
  def options(self, options, default=1):
    if options == None:
        raise Exception('Options is None')
    if len(options) == 0:
        raise Exception('Options has length 0')
    print('Which one would you like to choose?')
    for i, opt in enumerate(options):
        print(f'{i+1}) {opt}')
    x = input('')
    for i, option in enumerate(options):
        if x.lower() == option.lower():
            return i + 1
    try:
        x = int(x)
        if x >= 1 and x <= len(options):
            return x
        else:
            return default
    except:
        return default

  def accept(self, text):
    x = input(text)
    # check if there is yes or no
    x = x.lower()
    if x == 'yes' or x =='y':
        return True
    if x == 'no' or x =='n':
        return False
    try:
        x = int(x)
        if x == 1:
            return True
        return False
    except:
        return False

  # OFFLINE 
  # prepare path and images
  def delete_folder_content(self, folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

  def mkdir(self, dir):
    if not os.path.exists(dir):
      print(f'Making directory: {str(dir)}')
      os.makedirs(dir)
    
  def paths(self):
    self.starting = input('Enter the starting path: ')
    self.destination = input('Enter the destination path: ')
    self.mkdir(self.destination)
    n = input('Enter the number of the labels, default is 2 (0,1): ')
    self.labels = []
    try:
      n = int(n)
      if n>1:
        for i in range(n):
          self.labels.append(input('Enter label name '))
      else:
        n = 2
        self.labels = ['0', '1']
      
    except:
      n = 2
      self.labels = ['0', '1']
    
  def acquire_images(self,path):
    images = []
    types = ('*.png', '*.jpg', '*.jpeg')
    paths = []
    for typ in types:
        paths.extend(pathlib.Path(path).glob(typ))
    paths = paths[0:n_ims]
    for i in paths:
      im = cv2.imread(str(i))
      images.append(im)
    return images

  # transform images
  # assumption, starting point is rgb
  def modify_color(self, img):
    # HSV color
    if self.color == 2:
      img = skimage.color.rgb2hsv(img)
    # grayscale is default color
    if self.color != 1 and self.color != 2:
       img = skimage.color.rgb2gray(img)
    return img
  
  def strc(self, img):
    im_dim = np.shape(img)
    dim = (self.size, self.size, im_dim[2])
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img
  
  def get_max_and_index(self, l):
    value = max(l)
    for i, el in enumerate(l):
       if el == value:
          index = i
    return index, value
  
  def get_dimensions(self,height, width):
    list_size = []
    list_size.append(math.floor((self.size - height)/2))
    list_size.append(math.ceil((self.size - height)/2))
    list_size.append(math.floor((self.size - width)/2))
    list_size.append(math.ceil((self.size - width)/2))
    return list_size
  
  def fill(self, img):
    im_dim = np.shape(img)
    index, value = self.get_max_and_index(im_dim)
    if index:
      dim = (math.floor((im_dim[0]*self.size)/im_dim[1]), self.size)
    else:
      dim = (self.size, math.floor((im_dim[1]*self.size)/im_dim[0]))
    img = cv2.resize(img, (dim[1], dim[0]), interpolation = cv2.INTER_AREA)
    dimensions = img.shape
    tblr = self.get_dimensions(dimensions[0],dimensions[1])
    img = cv2.copyMakeBorder(img, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])
    return img
       
  def modify_size(self, img):
    if self.stretch:
      img = self.strc(img)
    else:
      img = self.fill(img)
    return img

  def modify(self, dataset, label):
     for i, img in enumerate(dataset):
        img = self.modify_size(img)
        img = self.modify_color(img)
        dest = self.destination + '/' + str(self.size) + '/' + self.opts[self.color-1]  + '/' + label
        self.mkdir(dest)
        im = Image.fromarray(np.uint8(img*255))
        im.save(dest + '/im' + str(i) + '.jpeg')
        #cv2.imwrite(dest + '/' + str(i) + '.jpg', img)

  def preferences(self):
    # get color and size and preferences
    self.opts = ['RGB', 'HSV', 'Greyscale']
    self.color = self.options(self.opts, default=3)
    size = input('Enter the size (default 33): ')
    try:
       self.size = int(size)
    except:
       self.size = 33
    self.stretch = self.accept('Do you want to stretch the image (default, fill with white pixels) ')
    
  

  def prepare(self):
    # prepare path and images
    self.paths()
    self.preferences()
    # transform images
    for label in self.labels:
      dataset = self.acquire_images(self.starting + '/' + label)
      self.modify(dataset, label)
    

  # ONLINE 
  def get_labels(self, path):
    dir_list = []
    for file in os.listdir(path):
        d = os.path.join(path, file)
        if os.path.isdir(d):
            dir_list.append(d)
    return dir_list

  def splitting(self, path, i, label):
    images = self.acquire_images(path + '/' + label)
    training = []
    test = []
    if len(images)>1:
      for image in images[0:int(len(images*self.proportion))]: 
        training.append((image, i))
      for image in images[int(len(images*self.proportion)):len(images)-1]:
         test.append((image, i))
    else:
      print(f'{path} is empty')
    self.TS.append((training))
    self.TestS.append((test))

  def build_dataset(self, paths):
    self.proportion = 0.85
    # for each path build a dataset
    # the dataset is divided into training and test set
    self.TS = []
    self.TestS = []

    for path in paths:
      labels = self.get_labels(path)
      for i, label in enumerate(labels):
        self.splitting(path, i, label)
    

