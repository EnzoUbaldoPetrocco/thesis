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

size = 100

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
    for i in range(300):
      index = random.randint(0,len(self.chinese)-1)
      temp_chin = self.chinese[index]
      temp_chin_cat = self.chinese_categories[index]
      self.chinese.pop(index)
      self.chinese.append(temp_chin)
      self.chinese_categories.pop(index)
      self.chinese_categories.append(temp_chin_cat)
    for i in range(300):
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
    file_name = "../accese vs spente.zip"
  # opening the zip file in READ mode
    with zipfile.ZipFile(file_name, 'r') as zip:
      zip.extractall('../')
      print('Done!')
    self.chinese = []
    self.chinese_categories = []
    self.french = []
    self.french_categories = []
    self.fill_chinese()
    self.fill_french()
    random.seed(time.time_ns())
    self.mix()
    self.mix_mixed_ds()
    self.size = size
