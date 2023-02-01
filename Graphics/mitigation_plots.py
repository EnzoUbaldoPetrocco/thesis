#! /usr/bin/env python3
from graphs import Histograms, Mitigation
import numpy as np
from matplotlib import pyplot as plt

chinese_data = {
    'lambda0': {
        'Accuracy_main': 75.62913907284768,
        'Accuracy_second': 49.62472406181017
    },
    'lambda': {
        'Accuracy_main': [74.96688741721854,75.25386313465783,75.78366445916114,75.09933774834437,74.083885209713,
        74.85651214128035,76.09271523178808,74.39293598233996,76.24724061810156,75.20971302428256,
        75.2980132450331,75.01103752759381,75.78366445916114,75.96026490066224,76.18101545253862,
        76.42384105960265,76.02649006622516,75.93818984547461,75.18763796909494,75.89403973509934,
        76.0485651214128,75.05518763796908,75.84988962472406,76.02649006622516 , 75.58498896247241 ],
        'Accuracy_second': [52.67108167770419,51.037527593818986,53.33333333333333,52.86975717439293,54.52538631346577,
        55.364238410596016,56.60044150110375,58.03532008830021,56.99779249448122,58.54304635761589,
        60.19867549668875,58.631346578366454,59.116997792494494,61.03752759381898,60.64017660044149,
        61.12582781456953,60.22075055187638,60.59602649006622,59.801324503311264,59.66887417218543,
        61.78807947019867,61.10375275938189,59.933774834437095,61.05960264900662 ,60.59602649006622 ]
    }
}

french_data = {
    'lambda0': {
        'Accuracy_main': 58.189845474613676,
        'Accuracy_second' : 50.0
    },
    'lambda': {
        'Accuracy_main': [56.556291390728475,58.18984547461368,58.12362030905078,58.278145695364216,
        58.27814569536423,58.587196467991156,57.63796909492274,57.262693156732894,56.975717439293604,
        56.86534216335541,56.887417218543035,53.973509933774835,56.070640176600435,55.67328918322297,
        56.90949227373069,52.958057395143484,55.47461368653423,53.95143487858719,53.70860927152317,
        52.22958057395143,53.841059602649,53.81898454746136,54.90066225165563,54.768211920529794 ,
        53.7306843267108],
        'Accuracy_second': [52.82560706401766,52.8476821192053,53.024282560706396,53.04635761589403,
        55.71743929359823,55.58498896247241,55.209713024282564,56.688741721854306,58.85209713024281,
        61.192052980132445,61.25827814569536,60.860927152317885,62.56070640176601,61.85430463576158,
        63.28918322295805,62.384105960264904,63.3112582781457,63.200883002207505,63.53200883002208,
        63.245033112582774,62.980132450331126,63.15673289183222,63.50993377483444,63.112582781456965 ,
        62.671081677704194 ]
    }
}

############################################
########    ACCURACIES #####################

#x = np.logspace(0,2,25)
x = np.linspace(0,100,25)
'''
chinese_mitigation = Mitigation(acc_main_class=chinese_data['lambda']['Accuracy_main'], acc_second_class=chinese_data['lambda']['Accuracy_second'],x=x,
fig_title='Chinese Mitigation', titles=["Chinese Accuracy", "French Accuracy", "3D Plot", "Metric Accuracy"] )
french_mitigation = Mitigation(acc_main_class=french_data['lambda']['Accuracy_main'], acc_second_class=french_data['lambda']['Accuracy_second'],x=x,
fig_title='French Mitigation', titles=["French Accuracy", "Chinese Accuracy", "3D Plot", "Metric Accuracy"] )
chin_tit= {'main': 'Chinese Accuracies', 'second': 'French Accuracies', 'metric': 'Mean Accuracies'}
fren_tit= {'second': 'Chinese Accuracies', 'main': 'French Accuracies', 'metric': 'Mean Accuracies'}
chinese_mitigation.plot_main()
chinese_mitigation.plot_second()
chinese_mitigation.plot_metric()
chinese_mitigation.plot_3d()

french_mitigation.plot_main()
french_mitigation.plot_second()
french_mitigation.plot_metric()
french_mitigation.plot_3d()

#chinese_mitigation.plot_together()
#french_mitigation.plot_together()
chinese_mitigation.plot_single(chin_tit)
french_mitigation.plot_single(fren_tit)

###########################################
## First metric: OPTIMAL LAMBDA IS MINORITY CLASS MAX
title = "Difference between Lambda=0 and Lambda*=max(minority class)= "
#### CHINESE #####
max_value = max(chinese_data['lambda']['Accuracy_second'])
max_index = chinese_data['lambda']['Accuracy_second'].index(max_value)
chinese_labels = ["Lambda 0 Chinese", "Lambda 0 French", "Lambda Grid " + str(max_index) + " Chinese", "Lambda Grid " + str(max_index)  + " French"]
chinese_hist = Histograms(chinese_labels, [chinese_data['lambda0']['Accuracy_main'], chinese_data['lambda0']['Accuracy_second'], 
chinese_data['lambda']['Accuracy_main'][max_index], chinese_data['lambda']['Accuracy_second'][max_index]],'Accuracy',title=title + str(x[max_index])[0:4] )
#### FRENCH #####
max_value = max(french_data['lambda']['Accuracy_second'])
max_index = french_data['lambda']['Accuracy_second'].index(max_value)
french_labels = ["Lambda 0 Chinese", "Lambda 0 French", "Lambda " + str(max_index) + " Chinese", "Lambda " + str(max_index) + " French"]
french_hist = Histograms(french_labels, [french_data['lambda0']['Accuracy_second'], french_data['lambda0']['Accuracy_main'], 
french_data['lambda']['Accuracy_second'][max_index], french_data['lambda']['Accuracy_main'][max_index]],'Accuracy', title=title + str(x[max_index])[0:4])

chinese_hist.plot()
french_hist.plot()

## Second metric: OPTIMAL LAMBDA IS OPTIMAL VALUE OF (x+y)/2
title = "Difference between Lambda=0 and Lambda*=max(minority+majority)/2= "
#### CHINESE ####
chin_func = []
for i in range(0,len(chinese_data['lambda']['Accuracy_main'])):
    k = (chinese_data['lambda']['Accuracy_main'][i]+chinese_data['lambda']['Accuracy_second'][i])/2
    chin_func.append(k)
max_value = max(chin_func)
max_index = chin_func.index(max_value)
chinese_labels = ["Lambda 0 Chinese", "Lambda 0 French", "Lambda Grid " + str(max_index) + " Chinese", "Lambda Grid " + str(max_index)  + " French"]
chinese_hist = Histograms(chinese_labels, [chinese_data['lambda0']['Accuracy_main'], chinese_data['lambda0']['Accuracy_second'], 
chinese_data['lambda']['Accuracy_main'][max_index], chinese_data['lambda']['Accuracy_second'][max_index]],'Accuracy',title=title +str(x[max_index])[0:4] )
#### FRENCH #####

#### FRENCH ####
fren_func = []
for i in range(0,len(french_data['lambda']['Accuracy_main'])):
    k = (french_data['lambda']['Accuracy_main'][i]+french_data['lambda']['Accuracy_second'][i])/2
    fren_func.append(k)
max_value = max(fren_func)
max_index = fren_func.index(max_value)
french_labels = ["Lambda 0 Chinese", "Lambda 0 French", "Lambda Grid " + str(max_index) + " Chinese", "Lambda Grid " + str(max_index)  + " French"]
french_hist = Histograms(french_labels, [french_data['lambda0']['Accuracy_second'], french_data['lambda0']['Accuracy_main'], 
french_data['lambda']['Accuracy_second'][max_index], french_data['lambda']['Accuracy_main'][max_index]],'Accuracy', title=title + str(x[max_index])[0:4])

chinese_hist.plot()
french_hist.plot()
'''

print('CHINESE DROPS')
for i in range(0,25):
    drop = chinese_data['lambda']['Accuracy_main'][i] - french_data['lambda']['Accuracy_second'][i]
    print("Drop for lambda in grid(" + str(i) + "):" + str(drop))

print('FRENCH DROPS')
for i in range(0,25):
    drop = french_data['lambda']['Accuracy_main'][i] - chinese_data['lambda']['Accuracy_second'][i]
    print("Drop for lambda in grid(" + str(i) + "):" + str(drop))

########################################################
##############  DROPS ###############################
chindrop = []
for i in range(0,25):
    k = chinese_data['lambda']['Accuracy_main'][i] - french_data['lambda']['Accuracy_second'][i]
    chindrop.append(k)
frendrop = []
for i in range(0,25):
    k = french_data['lambda']['Accuracy_main'][i] - chinese_data['lambda']['Accuracy_second'][i]
    frendrop.append(k)
tick_points = 13
ticks_label = []        
logspac= np.logspace(-2,2,tick_points)
for i in range(0,tick_points):
    ticks_label.append(str(logspac[i])[0:6])
ticks = np.linspace(0, 100, tick_points)
fig, ax = plt.subplots()
ax.set_title('Chinese Drops')
ax.legend(["Drop of Accuracy"])
ax.set_xlabel("Lambda")
ax.set_ylabel("Drop of Accuracy from Chinese to French model")
plt.xticks(ticks=ticks, labels=ticks_label)
ax.plot(x, chindrop, linewidth=2.0)
plt.show()
fig, ax = plt.subplots()
ax.set_title('French Drops')
ax.set_xlabel("Lambda")
ax.set_ylabel("Drop of Accuracy from French to Chinese model")
ax.legend(["Drop of Accuracy"])
plt.xticks(ticks=ticks, labels=ticks_label)
ax.plot(x, frendrop, linewidth=2.0)
plt.show()

mindrop = []
for i in range(0,25):
    k = (chindrop[i]+frendrop[i])/2
    mindrop.append(k)
fig, ax = plt.subplots()
ax.set_title('Mean Drops')
ax.legend(["Drop of Accuracy"])
ax.set_ylabel("Mean Drop of Accuracy")
ax.set_xlabel("Lambda")
plt.axhline(y = 5.15, color = 'r', linestyle = '-')
plt.xticks(ticks=ticks, labels=ticks_label)
ax.plot(x, mindrop, linewidth=2.0)
plt.show()