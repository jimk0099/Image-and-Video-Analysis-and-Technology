from __future__ import absolute_import, division, print_function, unicode_literals # legacy compatibility

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# select from from_list elements with index in index_list
def select_from_list(from_list, index_list):
  filtered_list= [from_list[i] for i in index_list]
  return(filtered_list)

# append in filtered_list the index of each element of unfiltered_list if it exists in in target_list
def get_ds_index(unfiltered_list, target_list):
  index = 0
  filtered_list=[]
  for i_ in unfiltered_list:
    if i_[0] in target_list:
      filtered_list.append(index)
    index += 1
  return(filtered_list)

# load the entire dataset
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

(x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

print(x_train_all.shape)

"""Η κάθε ομάδα θα δουλέψει με διαφορετικό υποσύνολο του dataset.
Στο επόμενο κελί, αντικαταστήστε την τιμή της μεταβλητής `team_seed` με τον αριθμό που αντιστοιχεί στην ομάδας σας. 
"""

# REPLACE WITH YOUR TEAM NUMBER
team_seed = 9

# select from CIFAR100 20 classes
cifar100_classes_url = "https://pastebin.com/raw/nzE1n98V"

"""Δημιουργούμε το μοναδικό dataset της ομάδας μας:"""

team_classes = pd.read_csv(cifar100_classes_url, sep=',', header=None)
CIFAR100_LABELS_LIST = pd.read_csv('https://pastebin.com/raw/qgDaNggt', sep=',', header=None).astype(str).values.tolist()[0]

our_index = team_classes.iloc[team_seed,:].values.tolist()
our_classes = select_from_list(CIFAR100_LABELS_LIST, our_index)
train_index = get_ds_index(y_train_all, our_index)
test_index = get_ds_index(y_test_all, our_index)

x_train_ds = np.asarray(select_from_list(x_train_all, train_index))
y_train_ds = np.asarray(select_from_list(y_train_all, train_index))
x_test_ds = np.asarray(select_from_list(x_test_all, test_index))
y_test_ds = np.asarray(select_from_list(y_test_all, test_index))

#print(our_index)
plt.imshow(x_train_ds[0])
print(y_train_ds[0])

# print our classes
print(our_classes)
print(our_index)
print((y_train_ds))

print(x_train_ds.shape)

# fix mapping [0, numclasses-1] -> [0, 19]
d = {}
for i in range(20):
  d[our_index[i]] = i # new key, so add
print(d)

# apply mapping to y_train_ds and make the z_train_ds matrix

z_train_ds = y_train_ds.copy()
z_test_ds = y_test_ds.copy()

for k in range(len(y_train_ds)):
  z_train_ds[k][0] = d[y_train_ds[k][0]]

for k in range(len(y_test_ds)):
  z_test_ds[k][0] = d[y_test_ds[k][0]]

# get (train) dataset dimensions
data_size, img_rows, img_cols, img_channels = x_train_ds.shape

# set validation set percentage (wrt the training set size)
validation_percentage = 0.15
val_size = round(validation_percentage * data_size)

# Reserve val_size samples for validation and normalize all values
x_val = x_train_ds[-val_size:]/255
y_val = z_train_ds[-val_size:]
x_train = x_train_ds[:-val_size]/255
y_train = z_train_ds[:-val_size]
x_test = x_test_ds/255
y_test = z_test_ds

# summarize loaded dataset
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Validation: X=%s, y=%s' % (x_val.shape, y_val.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# get class label from class index
def class_label_from_index(fine_category):
  return(CIFAR100_LABELS_LIST[fine_category.item(0)])

# plot first few images
plt.figure(figsize=(6, 7))
for i in range(9):
	# define subplot
  plt.subplot(330 + 1 + i).set_title(class_label_from_index(y_train_ds[i]))
	# plot raw pixel data
  plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
  #show the figure
plt.show()