#! /usr/bin/env python3
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

def get_dimensions(height, width):
  list_size = []
  list_size.append(math.floor((2500 - height)/2))
  list_size.append(math.ceil((2500 - height)/2))
  list_size.append(math.floor((1500 - width)/2))
  list_size.append(math.ceil((1500 - width)/2))
  return list_size

def fill_chinese():
  global chinese, chinese_categories
  path = '../accese vs spente/cinesi/'
  #paths_chin_off = pathlib.Path(path).glob('*.png')
  types = ('*.png', '*.jpg', '*.jpeg') # the tuple of file types
  paths_chin_off = []
  for files in types:
      paths_chin_off.extend(pathlib.Path(path).glob(files))
  ds_sorted_chin_off = sorted([x for x in paths_chin_off])

  example_im = cv2.imread(str(ds_sorted_chin_off[1]))
  imgplot = plt.imshow(example_im)
  dimensions = example_im.shape
  tblr = get_dimensions(dimensions[0],dimensions[1])
  plt.show()
  resized_image = cv2.copyMakeBorder(example_im, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])
  example_im = rgb2gray(example_im)
  imgplot = plt.imshow(example_im)
  plt.show()
  imgplot = plt.imshow(resized_image)
  plt.show()
  #dimensions = (height, width)
  


  for i in ds_sorted_chin_off:
    #im = np.array(imageio.imread(str(i)))
    im = cv2.imread(str(i))
    im = rgb2gray(im)
    #im = rgb2gray(im)
    #im = resize(im, (200,200))
    chinese.append(im.flatten())
    chinese_categories.append(0)
  path = '../accese vs spente/cinesi accese/'
  paths_chin_on = []
  for files in types:
      paths_chin_on.extend(pathlib.Path(path).glob(files))
  ds_sorted_chin_on = sorted([x for x in paths_chin_on])
  for i in ds_sorted_chin_on:
    #im = np.array(imageio.imread(str(i)))
    im = cv2.imread(str(i))
    im = rgb2gray(im)
    #im = resize(im, (200,200))
    chinese.append(im.flatten())
    chinese_categories.append(1)
  return chinese


def fill_french():
  global french, french_categories
  path = '../accese vs spente/francesi_accese/'
  types = ('*.png', '*.jpg', '*.jpeg')
  paths_fren_on = []
  for files in types:
      paths_fren_on.extend(pathlib.Path(path).glob(files))
  ds_sorted_fren_on = sorted([x for x in paths_fren_on])
  for i in ds_sorted_fren_on:
    #im = np.array(imageio.imread(str(i)))
    im = cv2.imread(str(i))
    im = rgb2gray(im)
    #im = resize(im, (200,200))
    french.append(im.flatten())
    french_categories.append(1)
  path = '../accese vs spente/francesi/'
  paths_fren_off = []
  for files in types:
      paths_fren_off.extend(pathlib.Path(path).glob(files))
  ds_sorted_fren_off = sorted([x for x in paths_fren_off])
  for i in ds_sorted_fren_off:
    #im = np.array(imageio.imread(str(i)))
    im = cv2.imread(str(i))
    #im = resize(im, (200,200))
    im = rgb2gray(im)
    french.append(im.flatten())
    french_categories.append(0)
  return french

chinese = fill_chinese()
#french = fill_french()

#img = chinese[]
#print(img)
#imgplot = plt.imshow(img)
#plt.show()
#print(french)
#print(chinese)
