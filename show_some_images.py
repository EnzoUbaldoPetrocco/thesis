#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import manipulating_images
from math import floor
import random
import time
############################################################
############### READ DATA ##################################
itd = manipulating_images.ImagesToData()

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
#######################PRINT SOME IMAGE VALUE#######################
for i in FX:
    for j in i:
        print(j)
####################################################################
###################### PLOT IMAGE ##################################
print('PLOT IMAGE RANDOM')
random.seed(int(time.time_ns()))
plt.figure()
plt.imshow(np.reshape(CX[random.randint(0,len(CX))], (itd.size,itd.size)))
plt.show()
'''plt.figure()
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
plt.show()'''