#! /usr/bin/env python3
import io
import os
import zipfile
import numpy as np
import imageio
import pathlib
from PIL import Image
from skimage.color import rgb2gray


file_name = "../accese vs spente.zip"
# opening the zip file in READ mode
with zipfile.ZipFile(file_name, 'r') as zip:
    zip.extractall('../')
    print('Done!')


chinese = []
chinese_categories = []
french = []
french_categories = []


def fill_chinese():
  global chinese, chinese_categories
  path = './accese vs spente/cinesi/'
  #paths_chin_off = pathlib.Path(path).glob('*.png')
  types = ('*.png', '*.jpg', '*.jpeg') # the tuple of file types
  paths_chin_off = []
  for files in types:
      paths_chin_off.extend(pathlib.Path(path).glob(files))
  ds_sorted_chin_off = sorted([x for x in paths_chin_off])
  ims_chin_off = []
  ims_chin_categories = []
  for i in ds_sorted_chin_off:
    im = np.array(imageio.imread(str(i)))
    #im = resize(im, (200,200))
    chinese.append(im.flatten())
    chinese_categories.append(0)
  path = './accese vs spente/cinesi accese/'
  paths_chin_on = []
  for files in types:
      paths_chin_on.extend(pathlib.Path(path).glob(files))
  ds_sorted_chin_on = sorted([x for x in paths_chin_on])
  ims_chin_on = []
  for i in ds_sorted_chin_on:
    im = np.array(imageio.imread(str(i)))
    #im = resize(im, (200,200))
    chinese.append(im.flatten())
    chinese_categories.append(1)


def fill_french():
  global french, french_categories
  path = './accese vs spente/francesi_accese/'
  types = ('*.png', '*.jpg', '*.jpeg')
  paths_fren_on = []
  for files in types:
      paths_fren_on.extend(pathlib.Path(path).glob(files))
  ds_sorted_fren_on = sorted([x for x in paths_fren_on])
  ims_fren_on = []
  ims_fren_categories = []
  for i in ds_sorted_fren_on:
    im = np.array(imageio.imread(str(i)))
    #im = resize(im, (200,200))
    french.append(im.flatten())
    french_categories.append(1)
  path = './accese vs spente/francesi/'
  paths_fren_off = []
  for files in types:
      paths_fren_off.extend(pathlib.Path(path).glob(files))
  ds_sorted_fren_off = sorted([x for x in paths_fren_off])
  ims_fren_off = []
  for i in ds_sorted_fren_off:
    im = np.array(imageio.imread(str(i)))
    #im = resize(im, (200,200))
    french.append(im.flatten())
    french_categories.append(0)


