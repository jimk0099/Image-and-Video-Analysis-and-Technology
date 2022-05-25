# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:26:33 2022

@author: Admin
"""

from intro import x_train, y_train, x_test, y_test

import tensorflow as tf

   
def AlexNet():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=2,
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(20)
    ])


X = tf.random.uniform((1, 32, 32, 3))
for layer in AlexNet().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

AlexNet_model_1 = AlexNet()
AlexNet_model_1.build((1, 32, 32, 3))           # each image is size 32x32 and has 3 channels (rgb)

AlexNet_model_2 = AlexNet()
AlexNet_model_2.build((1, 32, 32, 3)) 

AlexNet_model_1.summary()


# Train Model
AlexNet_model_1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

AlexNet_history_1 = AlexNet_model_1.fit(x_train, y_train, epochs=30, batch_size=128,
                    validation_data=(x_test, y_test))

AlexNet_model_1.save('AlexNet_model_1.h5')


AlexNet_model_2.compile(optimizer='RMSprop',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

AlexNet_history_2 = AlexNet_model_2.fit(x_train, y_train, epochs=30, batch_size=64,
                    validation_data=(x_test, y_test))

AlexNet_model_2.save('AlexNet_model_2.h5')


