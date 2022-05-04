#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import manipulating_images
from math import floor
from sklearn.metrics import confusion_matrix
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import time
from torch.autograd import Variable

###########################################################
##################### DEVICE ##############################
print('DEVICE')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))

############################################################
############### READ DATA ##################################
print('READ DATA')
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
###################### PLOT IMAGE ##################################
print('PLOT IMAGE')
plt.figure()
plt.imshow(np.reshape(CX[30], (itd.size,itd.size)))
plt.show()
####################################################################
##################### LOADERS ######################################
print('LOADERS')
loaders = {
    'train' : DataLoader(CX,
                         batch_size = 100,
                         shuffle = True,
                         num_workers= 1),
    'test'  : DataLoader(CXT,
                         batch_size = 100,
                         shuffle = True,
                         num_workers= 1)
}
####################################################################
##################### CNN ##########################################
print('CNN CLASS')
class CNN(nn.Module):
  def __init__(self):
    super(CNN,self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(1,16,5,1,2),
        nn.ReLU(),
        nn.MaxPool2d(2))
    self.conv2 = nn.Sequential(
        nn.Conv2d(16,32,5,1,2),
        nn.ReLU(),
        nn.MaxPool2d(2))
    self.out = nn.Linear(32*7*7,10)
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0),-1)
    x = self.out(x)
    return x
cnn = CNN().to(device)
###################################################################
##################### LOSS FUNCTION ###############################
print('LOSS FUNCTION')
loss_func = nn.CrossEntropyLoss()

###################################################################
####################### OPTIMIZER #################################
print('OPTIMIZER')
optimizer = optim.Adam(cnn.parameters(),
                      lr = 0.01)

###################################################################
####################### TRAINING ##################################
print('TRAINING')
def train(num_epoch, cnn, loaders):
  cnn.train()
  t = time.time()
  for epoch in range(num_epoch):
    for i, (img, labels) in enumerate(loaders['train']):
      x = Variable(img).to(device)
      y = Variable(labels).to(device)
      output = cnn(x)
      loss = loss_func(output,y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if (i+1)%100 == 0:
        elapsed = time.time() - t 
        print('Epoch: [{}/{}] - Batch: [{}/{}] - Loss: {:.4f} - Time: {:.1f}'.
              format(epoch+1,num_epoch,i+1,len(loaders['train']),loss.item(),elapsed))
        t = time.time()
num_epoch = 10
train(num_epoch,cnn,loaders)

###################################################################
####################### TEST ######################################
print('TEST')
actual_y_test = []
predictes_y_test = []
cnn.to('cpu')
with torch.no_grad():
  for (images, labels) in loaders['test']:
    test_output = cnn(images)
    pred_y = torch.max(test_output,1)[1].data.squeeze()
    actual_y_test += labels.tolist()
    predictes_y_test += pred_y.tolist()
print(confusion_matrix(actual_y_test,predictes_y_test))

