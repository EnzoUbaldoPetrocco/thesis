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
from torch import randint
import os, shutil


size = 200
total_n_images = 469

class ImagesToData:

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

  def get_dimensions(self,height, width):
    list_size = []
    list_size.append(math.floor((size - height)/2))
    list_size.append(math.ceil((size - height)/2))
    list_size.append(math.floor((size - width)/2))
    list_size.append(math.ceil((size - width)/2))
    return list_size

  def manage_size(self,im):
    dimensions = im.shape
    im_try = im
    while dimensions[0]>size or dimensions[1]>size:
      width = int(im.shape[1] * 0.9)
      height = int(im.shape[0] * 0.9)
      dim = (width, height)
      try:
        im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA )
      except:
        print(dim)
        plt.figure()
        plt.imshow(im_try)
        plt.show()
      dimensions = im.shape
    return im

  def created_dir(self, dir):
    os.mkdir(dir)

  def create_directories(self):
    size_path = '../' + str(self.size)
    os.mkdir(size_path)
    chinese_path = '../' + str(self.size) + '/cinesi'
    os.mkdir(chinese_path)
    chinese_on_path = '../' + str(self.size) + '/cinesi accese'
    os.mkdir(chinese_on_path)
    french_on_path = '../' + str(self.size) + '/francesi accese'
    os.mkdir(french_on_path)
    french_path = '../' + str(self.size) + '/francesi'
    os.mkdir(french_path)
  
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
    for typ in types:
        paths.extend(pathlib.Path(path).glob(typ))
    #sorted_ima = sorted([x for x in paths])
    for i in paths:
      im = cv2.imread(str(i))
      im = self.modify_images(im)
      images.append(im)
    return images

  def acquire_images(self,path):
    images = []
    types = ('*.png', '*.jpg', '*.jpeg')
    paths = []
    for typ in types:
        paths.extend(pathlib.Path(path).glob(typ))
    #sorted_ima = sorted([x for x in paths])
    for i in paths:
      im = cv2.imread(str(i))
      im = self.modify_images(im)
      images.append(im)
    return images

  def save_images(self, list, path):
    for i in range(len(list)):
      im = numpy.reshape(list[i], (self.size,self.size))
      im = Image.fromarray(numpy.uint8(im*255))
      im.save(path + '/im' + str(i) + '.jpeg')

  def mix_list(self, list):
    for i in range(999999):
      index = random.randint(0,len(list)-1)
      temp = list[index]
      list.pop(index)
      list.append(temp)
    return list

  def initial_routine(self, create_directory):
    file_name = "../accese vs spente.zip"
  # opening the zip file in READ mode
    with zipfile.ZipFile(file_name, 'r') as zip:
      zip.extractall('../')
      print('Done!')
    if create_directory:
      self.create_directories()
    random.seed(time.time_ns())
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
    self.chinese_off = self.acquire_images('../../' + str(self.size) + '/cinesi')
    self.chinese_on = self.acquire_images('../../' + str(self.size) + '/cinesi accese')
    self.french_off = self.acquire_images('../../' + str(self.size) + '/francesi')
    self.french_on = self.acquire_images('../../' + str(self.size) + '/francesi accese')
    self.mix_list(self.chinese_off)
    self.mix_list(self.chinese_on)
    self.mix_list(self.french_off)
    self.mix_list(self.french_on)

    self.divide_ds_FE()

    self.chinese = numpy.concatenate((self.chinese_off, self.chinese_on),axis=0)
    self.french = numpy.concatenate((self.french_off, self.french_on),axis=0)
    self.chinese_categories = numpy.concatenate(((numpy.ones(len(self.chinese_off))*(0)), numpy.ones(len(self.chinese_on))))
    self.french_categories = numpy.concatenate(((numpy.ones(len(self.french_off))*(0)), numpy.ones(len(self.french_on))))
    random.seed(time.time_ns())
    self.mix()
    self.prepare_ds()

  def divide_ds_FE(self):
    self.prop = 1/2
    
    base_path = '../../FE/' + self.dspath
    self.delete_folder_content('../../FE')
    try:
      self.created_dir(base_path)
      self.delete_folder_content(base_path)
    except:
      print('base path not existing')
    self.created_dir(base_path + '/chinese')
    self.created_dir(base_path + '/french')
    self.created_dir(base_path + '/chinese/' + 'cinesi')
    self.created_dir(base_path + '/chinese/' + 'cinesi accese')
    self.created_dir(base_path + '/french/'+ 'francesi')
    self.created_dir(base_path + '/french/'+ 'francesi accese')

    self.save_images(self.chinese_on[0:int(len(self.chinese_on)*self.prop)], base_path  + '/chinese/'+ 'cinesi')
    self.save_images(self.chinese_off[0:int(len(self.chinese_off)*self.prop)], base_path  + '/chinese/'+ 'cinesi accese')
    self.save_images(self.french_on[0:int(len(self.french_on)*self.prop)], base_path + '/french/' + 'francesi')
    self.save_images(self.french_off[0:int(len(self.french_off)*self.prop)], base_path  + '/french/'+ 'francesi accese')
    self.chinese_on = self.chinese_on[int(len(self.chinese_on)*self.prop):len(self.chinese_on)-1]
    self.chinese_off = self.chinese_off[int(len(self.chinese_off)*self.prop):len(self.chinese_off)-1]
    self.french_on = self.french_on[int(len(self.french_on)*self.prop):len(self.french_on)-1]
    self.french_off = self.french_off[int(len(self.french_off)*self.prop):len(self.french_off)-1]
    


  def mix(self):
    self.chinese = list(self.chinese)
    self.chinese_categories = list(self.chinese_categories)
    self.french = list(self.french)
    self.french_categories = list(self.french_categories)
    for i in range(999999):
      index = random.randint(0,len(self.chinese)-1)
      temp_chin = self.chinese[index]
      temp_chin_cat = self.chinese_categories[index]
      self.chinese.pop(index)
      self.chinese.append(temp_chin)
      self.chinese_categories.pop(index)
      self.chinese_categories.append(temp_chin_cat)
    for i in range(999999):
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

  def prepare_ds(self):
    ### Divisions
    first = int(270*(1-self.prop))
    second = int((270 + 105)*(1-self.prop))
    fin = int(total_n_images*(1-self.prop))
    

    self.chinese = list(self.chinese)
    self.french = list(self.french)
    self.chinese_categories = list(self.chinese_categories)
    self.french_categories = list(self.french_categories)

    self.CX = self.chinese[0 : first]
    self.CY = self.chinese_categories[0 : first]
    self.CXT  = self.chinese[first +1 : second]
    self.CYT = self.chinese_categories[first + 1 : second]
    self.MXT = self.chinese[second + 1 : fin]
    self.MYT = self.chinese_categories[second + 1 : fin]
    


    self.FX = self.french[0 : first]
    self.FY = self.french_categories[0 : first]
    self.FXT  = self.french[first + 1 : second]
    self.FYT = self.french_categories[first + 1 : second]
    self.MXT = numpy.concatenate((self.MXT, self.french[second : fin]), axis = 0)
    self.MYT = numpy.concatenate((self.MYT, self.french_categories[second : fin]), axis = 0)
    self.MX = numpy.concatenate((self.CX, self.FX), axis=0)
    self.MY = numpy.concatenate((self.CY, self.FY), axis=0)

    self.CX = numpy.array(self.CX)
    self.CY = numpy.array(self.CY)
    self.CXT = numpy.array(self.CXT)
    self.CYT = numpy.array(self.CYT)

    self.FX = numpy.array(self.FX)
    self.FY = numpy.array(self.FY)
    self.FXT = numpy.array(self.FXT)
    self.FYT = numpy.array(self.FYT)
    
    self.MX = numpy.array(self.MX)
    self.MY = numpy.array(self.MY)
    self.MXT = numpy.array(self.MXT)
    self.MYT = numpy.array(self.MYT)

    
    

  def little_mix(self):
    self.MCX = list(self.CX)
    self.MCY = list(self.CY)

    self.MFX = list(self.FX)
    self.MFY = list(self.FY)
    for i in range(25):
      index = random.randint(0,len(self.FX)-11)
      self.MCX.append(self.FX[index])
      self.MCY.append(self.FY[index])

    for i in range(25):
      index = random.randint(0,len(self.FX)-11)
      self.MFX.append(self.CX[index])
      self.MFY.append(self.CY[index])

    self.MCX = numpy.array(self.MCX)
    self.MCY = numpy.array(self.MCY)
    self.MFX = numpy.array(self.MFX)
    self.MFY = numpy.array(self.MFY)


    


  def __init__(self, initialize = False, create_directory = False, ds_selection = 'default'):
    self.dspath = ds_selection
    self.size = size
    self.chinese = []
    self.chinese_categories = []
    self.french = []
    self.french_categories = []
    self.mixed = []
    self.mixed_categories = []
    if initialize:
      self.initial_routine(create_directory)
    
    


itd = ImagesToData(False, True)
#itd.initial_routine()
