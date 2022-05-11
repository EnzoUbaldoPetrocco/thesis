#! /usr/bin/env python3

from turtle import width
import zipfile
import pathlib
import numpy
from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt
import math
import pandas as pd
import random
import time
import os
from PIL import Image

size = 35

class ImagesToData:

  def get_dimensions(self,height, width):
    list_size = []
    list_size.append(math.floor((size - height)/2))
    list_size.append(math.ceil((size - height)/2))
    list_size.append(math.floor((size - width)/2))
    list_size.append(math.ceil((size - width)/2))
    return list_size

  def manage_size(self,im):
    dimensions = im.shape
    while dimensions[0]>size or dimensions[1]>size:
      width = int(im.shape[1] * 0.9)
      height = int(im.shape[0] * 0.9)
      dim = (width, height)
      im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA )
      dimensions = im.shape
    return im

  def create_directories(self):
    size_path = '../' + str(self.size)
    os.mkdir(size_path)
    chinese_path = '../' + str(self.size) + '/cinesi'
    os.mkdir(chinese_path)
    chinese_on_path = '../' + str(self.size) + '/cinesi accese'
    os.mkdir(chinese_on_path)
    french_path = '../' + str(self.size) + '/francesi accese'
    os.mkdir(french_path)
    french_on_path = '../' + str(self.size) + '/francesi'
    os.mkdir(french_on_path)
  
  def modify_images(self, im):
    im = self.manage_size(im)
    dimensions = im.shape
    tblr = self.get_dimensions(dimensions[0],dimensions[1])
    im = cv2.copyMakeBorder(im, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])
    im = rgb2gray(im)
    im_obj = pd.DataFrame(im).to_numpy()
    return im_obj.flatten()

  def acquire_modify_images(self,path):
    images = []
    types = ('*.png', '*.jpg', '*.jpeg')
    paths = []
    for files in types:
        paths.extend(pathlib.Path(path).glob(files))
    sorted_ima = sorted([x for x in paths])
    for i in sorted_ima:
      im = cv2.imread(str(i))
      im = self.modify_images(im)
      images.append(im)
    return images

  def acquire_images(self,path):
    images = []
    types = ('*.png', '*.jpg', '*.jpeg')
    paths = []
    for files in types:
        paths.extend(pathlib.Path(path).glob(files))
    #sorted_ima = sorted([x for x in paths])
    for i in paths:
      im = cv2.imread(str(i))
      im = self.modify_images(im)
      images.append(im)
    return images

  def save_images(self, list, path):
    for i in range(350):
      im = numpy.reshape(list[i], (self.size,self.size))
      im = Image.fromarray(numpy.uint8(im*255))
      im.save(path + '/im' + str(i) + '.jpeg')

  def mix_list(self, list):
    for i in range(10000):
      index = random.randint(0,len(list)-1)
      temp = list[index]
      list.pop(index)
      list.append(temp)
    return list

  def initial_routine(self):
    file_name = "../accese vs spente.zip"
  # opening the zip file in READ mode
    with zipfile.ZipFile(file_name, 'r') as zip:
      zip.extractall('../')
      print('Done!')
    self.create_directories()
    self.chinese_off = self.acquire_modify_images('../accese vs spente/cinesi/')
    self.chinese_on = self.acquire_modify_images('../accese vs spente/cinesi accese/')
    self.french_off = self.acquire_modify_images('../accese vs spente/francesi/')
    self.french_on = self.acquire_modify_images('../accese vs spente/francesi accese/')
    self.chinese_off = self.mix_list(self.chinese_off)
    self.chinese_on = self.mix_list(self.chinese_on)
    self.french_off = self.mix_list(self.french_off)
    self.french_on = self.mix_list(self.french_on)
    self.save_images(self.chinese_off, '../' + str(self.size) + '/cinesi')
    self.save_images(self.chinese_on, '../' + str(self.size) + '/cinesi accese')
    self.save_images(self.french_off, '../' + str(self.size) + '/francesi')
    self.save_images(self.french_on, '../' + str(self.size) + '/francesi accese')

  def bf_ml(self):
    chinese_off = self.acquire_images('../' + str(self.size) + '/cinesi')
    chinese_on = self.acquire_images('../' + str(self.size) + '/cinesi accese')
    french_off = self.acquire_images('../' + str(self.size) + '/francesi')
    french_on = self.acquire_images('../' + str(self.size) + '/francesi accese')
    self.chinese = numpy.concatenate((chinese_off, chinese_on),axis=0)
    self.french = numpy.concatenate((chinese_off, chinese_on),axis=0)
    self.chinese_categories = numpy.concatenate(((numpy.ones(len(chinese_off))*(-1)), numpy.ones(len(chinese_on))))
    self.french_categories = numpy.concatenate(((numpy.ones(len(french_off))*(-1)), numpy.ones(len(french_on))))
    random.seed(time.time_ns())
    self.mix()
    self.mix_mixed_ds()

  def fill_chinese(self):
    global chinese, chinese_categories
    path = '../accese vs spente/cinesi/'
    #paths_chin_off = pathlib.Path(path).glob('*.png')
    types = ('*.png', '*.jpg', '*.jpeg') # the tuple of file types
    paths_chin_off = []
    for files in types:
        paths_chin_off.extend(pathlib.Path(path).glob(files))
    ds_sorted_chin_off = sorted([x for x in paths_chin_off])
    
    for i in ds_sorted_chin_off:
      im = cv2.imread(str(i))
      im = self.manage_size(im)
      dimensions = im.shape
      tblr = self.get_dimensions(dimensions[0],dimensions[1])
      im = cv2.copyMakeBorder(im, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])
      im = rgb2gray(im)
      im_obj = pd.DataFrame(im).to_numpy()
      self.chinese.append(im_obj.flatten())
      #chinese.append(im.flatten())
      self.chinese_categories.append(0)
    path = '../accese vs spente/cinesi accese/'
    paths_chin_on = []
    for files in types:
        paths_chin_on.extend(pathlib.Path(path).glob(files))
    ds_sorted_chin_on = sorted([x for x in paths_chin_on])
    
    for i in ds_sorted_chin_on:
      im = cv2.imread(str(i))
      im = self.manage_size(im)
      dimensions = im.shape
      tblr = self.get_dimensions(dimensions[0],dimensions[1])
      im = cv2.copyMakeBorder(im, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])
      im = rgb2gray(im)

      im_obj = pd.DataFrame(im).to_numpy()

      self.chinese.append(im_obj.flatten())
      self.chinese_categories.append(1)
    
    return self.chinese


  def fill_french(self):
    global french, french_categories
    path = '../accese vs spente/francesi accese/'
    types = ('*.png', '*.jpg', '*.jpeg')
    paths_fren_on = []
    for files in types:
        paths_fren_on.extend(pathlib.Path(path).glob(files))
    ds_sorted_fren_on = sorted([x for x in paths_fren_on])
    
    for i in ds_sorted_fren_on:
      im = cv2.imread(str(i))
      im = self.manage_size(im)
      dimensions = im.shape
      tblr = self.get_dimensions(dimensions[0],dimensions[1])
      im = cv2.copyMakeBorder(im, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])
      im = rgb2gray(im)
      im_obj = pd.DataFrame(im).to_numpy()
      self.french.append(im_obj.flatten())
      self.french_categories.append(1)
    path = '../accese vs spente/francesi/'
    paths_fren_off = []
    for files in types:
        paths_fren_off.extend(pathlib.Path(path).glob(files))
    ds_sorted_fren_off = sorted([x for x in paths_fren_off])
    
    for i in ds_sorted_fren_off:
      im = cv2.imread(str(i))
      im = self.manage_size(im)
      dimensions = im.shape
      tblr = self.get_dimensions(dimensions[0],dimensions[1])
      im = cv2.copyMakeBorder(im, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])
      im = rgb2gray(im)
      im_obj = pd.DataFrame(im).to_numpy()

      self.french.append(im_obj.flatten())
      self.french_categories.append(0)
    return self.french

  def mix(self):

    self.chinese = list(self.chinese)
    self.chinese_categories = list(self.chinese_categories)
    self.french = list(self.french)
    self.french_categories = list(self.french_categories)
    for i in range(10000):
      index = random.randint(0,len(self.chinese)-1)
      temp_chin = self.chinese[index]
      temp_chin_cat = self.chinese_categories[index]
      self.chinese.pop(index)
      self.chinese.append(temp_chin)
      self.chinese_categories.pop(index)
      self.chinese_categories.append(temp_chin_cat)
    for i in range(10000):
      index = random.randint(0,len(self.french)-1)
      temp_fren = self.french[index]
      temp_fren_cat = self.french_categories[index]
      self.french.pop(index)
      self.french.append(temp_fren)
      self.french_categories.pop(index)
      self.french_categories.append(temp_fren_cat)
    
    self.chinese = numpy.array(self.chinese)
    self.chinese_categories = numpy.array(self.chinese_categories)

    self.french = numpy.array(self.french)
    self.french_categories = numpy.array(self.french_categories)

  def mix_mixed_ds(self):

    self.mixed = numpy.concatenate((self.chinese, self.french), axis=0)
    self.mixed_categories = numpy.concatenate((self.chinese_categories, self.french_categories), axis = 0)

    #self.mixed = self.mixed.tolist()
    #self.mixed_categories = self.mixed_categories.tolist()

    self.mixed = list(self.mixed)
    self.mixed_categories = list(self.mixed_categories)
    for i in range(300):
      index = random.randint(0,len(self.french)-1)
      temp_mix = self.mixed[index]
      temp_mix_cat = self.mixed_categories[index]
      self.mixed.pop(index)
      self.mixed.append(temp_mix)
      self.mixed_categories.pop(index)
      self.mixed_categories.append(temp_mix_cat)

    self.mixed = numpy.array(self.mixed)
    self.mixed_categories = numpy.array(self.mixed_categories)

  def __init__(self):
    
    self.size = size
    self.chinese = []
    self.chinese_categories = []
    self.french = []
    self.french_categories = []
    self.mixed = []
    self.mixed_categories = []
    random.seed(7)
    


itd = ImagesToData()
itd.initial_routine()
