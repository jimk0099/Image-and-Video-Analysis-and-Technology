# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:32:48 2022

@author: Admin
"""

from intro import x_test, y_test

import tensorflow as tf

LeNet_2 = tf.keras.models.load_model('LeNet_model_2.h5')      # Load model

loss, acc = LeNet_2.evaluate(x_test, y_test, verbose=2)
print('LeNet model, accuracy: {:5.2f}%'.format(100 * acc))

