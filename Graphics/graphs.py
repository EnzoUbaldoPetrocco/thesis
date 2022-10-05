#! /usr/bin/env python3
from matplotlib import pyplot as plt
import numpy as np

## This program should be used for creating plots of some relevant results

class Mitigation:
    def __init__(self, acc_main_class, acc_second_class, x):
        self.accuracy_main_class = acc_main_class
        self.accuracy_second_class = acc_second_class
        self.x = x

    def plot_main(self):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.accuracy_main_class, linewidth=2.0)
        plt.show()

    def plot_second(self):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.accuracy_second_class, linewidth=2.0)
        plt.show()
    
    def plot_3d(self):
        ax = plt.axes(projection='3d')
        ax.set_title('Plot 3D',
             fontsize = 14)
        X, Y, Z = self.x, self.accuracy_main_class, self.accuracy_second_class
        ax.plot3D(X , Y, Z, 'gray')
        '''
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
        '''
        plt.show()

class Histograms:
    def __init__(self, labels, percentages, y_labels, title):
        self.labels = labels
        self.percentages = percentages
        self.y_labels = y_labels
        self.title = title

    def plot(self):
        fig, ax = plt.subplots()
        bar_colors = ['tab:red', 'tab:blue', 'tab:black', 'tab:orange']
        ax.bar(self.labels, self.percentages, label=self.labels, color=bar_colors[:len(self.labels)-1])
        ax.set_ylabel(self.y_labels)
        ax.set_title(self.title)
        plt.show()


rng = np.random.default_rng(12345)
acc_main_class = []
acc_second_class = []
for i in range(0,25):
    rfloat = rng.random()
    acc_main_class.append(75 + rfloat%10)
for i in range(0,25):
    rfloat = rng.random()
    acc_second_class.append(60 + rfloat%10)
x = np.logspace(0,2,25)
mit = Mitigation(acc_main_class, acc_second_class, x)
mit.plot_main()
mit.plot_second()
mit.plot_3d()

labels = ["chinese", "french"]
percentages = [60, 80]
y_labels = ["chin", "french"]
title = "Histogram"
hist = Histograms(labels, percentages, y_labels, title )
hist.plot()


