# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:39:07 2022

@author: Admin
"""

import matplotlib as plt
from LeNet import LeNet_history_1, LeNet_history_2

# Define plot function
def plot_accuracy(model_history_1, model_history_2, titles):

  epochs_1 = range(1, len(model_history_1.history['accuracy']) + 1)             # get number of epochs
  accuracy_1 = model_history_1.history['accuracy']
  val_accuracy_1 = model_history_1.history['val_accuracy']

  epochs_2 = range(1, len(model_history_2.history['accuracy']) + 1)
  accuracy_2 = model_history_2.history['accuracy']
  val_accuracy_2 = model_history_2.history['val_accuracy']


  fig, ax = plt.subplots(1, 2, figsize=(20, 7))

  ax[0].plot(epochs_1, accuracy_1, label='accuracy')
  ax[0].plot(epochs_1, val_accuracy_1, label='val_accuracy')
  ax[0].set_title(model_history_1)
  ax[0].set_xlabel('Epoch')
  ax[0].set_ylabel('Accuracy')
  ax[0].set_ylim(0, 1)
  ax[0].legend()
  ax[0].set_title(titles[0])

  ax[1].plot(epochs_2, accuracy_2, label='accuracy')
  ax[1].plot(epochs_2, val_accuracy_2, label='val_accuracy')
  ax[1].set_title(model_history_2)
  ax[1].set_xlabel('Epoch')
  ax[1].set_ylabel('Accuracy')
  ax[1].set_ylim(0, 1)
  ax[1].legend()
  ax[1].set_title(titles[1])
  
  
plot_accuracy(LeNet_history_1, LeNet_history_2, ['LeNet Model 1', 'LeNet Model 2'])