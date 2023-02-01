#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import manipulating_images_better
from math import floor
import random
import time
import cv2
############################################################
############### READ DATA ##################################
itd = manipulating_images_better.ImagesToData()
itd.bf_ml()

CX = itd.chinese[0:floor(len(itd.chinese)*0.7)]
CXT = itd.chinese[floor(len(itd.chinese)*0.7):len(itd.chinese)-1]
CY = itd.chinese_categories[0:floor(len(itd.chinese)*0.7)]
CYT = itd.chinese_categories[floor(len(itd.chinese)*0.7):len(itd.chinese)-1]

FX = itd.french[0:floor(len(itd.french)*0.7)]
FXT = itd.french[floor(len(itd.french)*0.7):len(itd.french)-1]
FY = itd.french_categories[0:floor(len(itd.french)*0.7)]
FYT = itd.french_categories[floor(len(itd.french)*0.7):len(itd.french)-1]

MX = itd.mixed[0:floor(len(itd.mixed)*0.7)]
MXT = itd.mixed[floor(len(itd.mixed)*0.7):len(itd.mixed)-1]
MY = itd.mixed_categories[0:floor(len(itd.mixed)*0.7)]
MYT = itd.mixed_categories[floor(len(itd.mixed)*0.7):len(itd.mixed)-1]

####################################################################
###################### PLOT IMAGE ##################################
print('PLOT IMAGE RANDOM')
random.seed(int(time.time_ns()))
fig = plt.figure(figsize=(1,3))
columns = 6
rows = 2
fig.suptitle("Dataset Images")
for i in range(1, columns*rows +1):
    if i <= 5:
        img = np.reshape(CX[random.randint(0,len(CX))], (itd.size,itd.size))
    else:
        img = np.reshape(FX[random.randint(0,len(FX))], (itd.size,itd.size))
    img = cv2.merge([img,img,img])
    ax = fig.add_subplot(rows, columns, i)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(img)
plt.show()
'''
plt.figure()
plt.imshow(np.reshape(CX[random.randint(0,len(CX))], (itd.size,itd.size)))
plt.show()
plt.figure()
plt.imshow(np.reshape(FX[random.randint(0,len(FX))], (itd.size,itd.size)))
plt.show()
plt.figure()
plt.imshow(np.reshape(MX[random.randint(0,len(MX))], (itd.size,itd.size)))
plt.show()
plt.figure()
plt.imshow(np.reshape(CXT[random.randint(0,len(CXT))], (itd.size,itd.size)))
plt.show()
plt.figure()
plt.imshow(np.reshape(FXT[random.randint(0,len(FXT))], (itd.size,itd.size)))
plt.show()
plt.figure()
plt.imshow(np.reshape(MXT[random.randint(0,len(MXT))], (itd.size,itd.size)))
plt.show()
'''