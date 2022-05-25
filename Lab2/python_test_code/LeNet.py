# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:00:41 2022

@author: Admin
"""

from intro import x_train, y_train, x_test, y_test

import tensorflow as tf


def main():
    pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here

    def LeNet():
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                                   padding='same'),
            tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                                   activation='sigmoid'),
            tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation='sigmoid'),
            tf.keras.layers.Dense(84, activation='sigmoid'),
            tf.keras.layers.Dense(20)])
    
    X = tf.random.uniform((1, 32, 32, 3))
    for layer in LeNet().layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: \t', X.shape)
    
    LeNet_model_1 = LeNet()
    LeNet_model_1.build((1, 32, 32, 3))           # each image is size 32x32 and has 3 channels (rgb)
    
    LeNet_model_2 = LeNet()
    LeNet_model_2.build((1, 32, 32, 3)) 
    
    LeNet_model_1.summary()
    
    
    
    # Train Model
    
    LeNet_model_1.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    LeNet_history_1 = LeNet_model_1.fit(x_train, y_train, epochs=50, batch_size=256,
                        validation_data=(x_test, y_test))
    
    LeNet_model_1.save('LeNet_model_1.h5')
    
    """* Δεύτερη υλοποίηση LeNet: optimizer='Nadam', epochs=50, batch_size=10"""
    
    LeNet_model_2.compile(optimizer='Nadam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    LeNet_history_2 = LeNet_model_2.fit(x_train, y_train, epochs=50, batch_size=10,
                        validation_data=(x_test, y_test))
    
    LeNet_model_2.save('LeNet_model_2.h5')
    
    
    main()