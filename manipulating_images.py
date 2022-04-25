#! /usr/bin/env python3
from configparser import Interpolation
import io
import os
import zipfile
import numpy as np
import imageio
import pathlib
from PIL import Image
from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import data
import math

file_name = "../accese vs spente.zip"
# opening the zip file in READ mode
with zipfile.ZipFile(file_name, 'r') as zip:
    zip.extractall('../')
    print('Done!')


chinese = []
chinese_categories = []
french = []
french_categories = []

path = '../chinese'
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)
path = '../french'
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

def get_dimensions(height, width):
  list_size = []
  list_size.append(math.floor((1000 - height)/2))
  list_size.append(math.ceil((1000 - height)/2))
  list_size.append(math.floor((500 - width)/2))
  list_size.append(math.ceil((500 - width)/2))
  return list_size

def manage_size(im):
  dimensions = im.shape
  
  if dimensions[0] > 3000 or dimensions[1]>1500:
    im = cv2.resize(im,(int(dimensions[0]*0.3),int(dimensions[1]*0.3)),interpolation = cv2.INTER_AREA )
    dimensions = im.shape
  if dimensions[0] > 2000 or dimensions[1]>1000:
    im = cv2.resize(im,(int(dimensions[0]*0.3),int(dimensions[1]*0.3)),interpolation = cv2.INTER_AREA )
    dimensions = im.shape
  if dimensions[0] > 1000 or dimensions[1]>500:
    im = cv2.resize(im,(int(dimensions[0]*0.3),int(dimensions[1]*0.3)),interpolation = cv2.INTER_AREA )
    
  return im

def fill_chinese():
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
    im = manage_size(im)
    dimensions = im.shape
    tblr = get_dimensions(dimensions[0],dimensions[1])
    im = cv2.copyMakeBorder(im, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])

    im = rgb2gray(im)
    chinese.append(im.flatten())
    chinese_categories.append(0)
  path = '../accese vs spente/cinesi accese/'
  paths_chin_on = []
  for files in types:
      paths_chin_on.extend(pathlib.Path(path).glob(files))
  ds_sorted_chin_on = sorted([x for x in paths_chin_on])
  for i in ds_sorted_chin_on:
    im = cv2.imread(str(i))
    im = manage_size(im)
    dimensions = im.shape
    tblr = get_dimensions(dimensions[0],dimensions[1])
    im = cv2.copyMakeBorder(im, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])
    im = rgb2gray(im)
    chinese.append(im.flatten())
    chinese_categories.append(1)
  return chinese


chinese = fill_chinese()
print('finished')

def fill_french():
  global french, french_categories
  path = '../accese vs spente/francesi_accese/'
  types = ('*.png', '*.jpg', '*.jpeg')
  paths_fren_on = []
  for files in types:
      paths_fren_on.extend(pathlib.Path(path).glob(files))
  ds_sorted_fren_on = sorted([x for x in paths_fren_on])
  for i in ds_sorted_fren_on:
    im = cv2.imread(str(i))
    im = manage_size(im)
    dimensions = im.shape
    tblr = get_dimensions(dimensions[0],dimensions[1])
    im = cv2.copyMakeBorder(im, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])
    im = rgb2gray(im)
    french.append(im.flatten())
    french_categories.append(1)
  path = '../accese vs spente/francesi/'
  paths_fren_off = []
  for files in types:
      paths_fren_off.extend(pathlib.Path(path).glob(files))
  ds_sorted_fren_off = sorted([x for x in paths_fren_off])
  for i in ds_sorted_fren_off:
    im = cv2.imread(str(i))
    im = manage_size(im)
    dimensions = im.shape
    tblr = get_dimensions(dimensions[0],dimensions[1])
    im = cv2.copyMakeBorder(im, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])
    im = rgb2gray(im)
    french.append(im.flatten())
    french_categories.append(0)
  return french

#french = fill_french()

#imgplot = plt.imshow(img)
#plt.show()
#print(french)
#print(chinese)
