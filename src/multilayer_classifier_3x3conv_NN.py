# -*- coding: utf-8 -*-
# Creator: Ryan Vansickle
# Co-Creator: Chau Pham
# Referenced https://github.com/Rajsoni03/neuralplot for visualizing keras model in NN
# Input: Keras Model 
# Output: 3D neural net visualization of keras model

# THIS ONLY WORKS IN GOOGLE COLAB ONLY NOT VSCODE
"""multilayer_classifier_3x3conv_NN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QU_8cNbrnl5h601FUuAA6VA8uyecEZ2L?usp=sharing

# Neural Net Visualization

## Install neuralplot library
"""
# For google colab only
# !pip install neuralplot

"""## Import Libraries"""

from neuralplot import ModelPlot
import tensorflow as tf
import numpy as np

# Commented out IPython magic to ensure Python compatibility.
# Uncomment while using Colab.
# %matplotlib inline

"""## Creating Model"""

input_layer = tf.keras.Input(shape=(32, 32, 3))

# Preprocessing and initial convolutional layer
x = tf.keras.layers.RandomRotation(0.1)(input_layer)
x = tf.keras.layers.Rescaling(1. / 255)(x)
x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)

previous_block_activation = x  # Save the current state of the network

# First block
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.SeparableConv2D(256, 3, padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.SeparableConv2D(256, 3, padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
residual = tf.keras.layers.Conv2D(256, 1, strides=2, padding="same")(previous_block_activation)
x = tf.keras.layers.add([x, residual])  # Add back residual (note: missing in your original)
previous_block_activation = x  # Update the activation

# Subsequent blocks (omitted for brevity, but follow the same pattern as the first block)

# Final layers and model creation
x = tf.keras.layers.SeparableConv2D(1024, 3, padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

model = tf.keras.models.Model(inputs=input_layer, outputs=outputs)

model.summary()

modelplot = ModelPlot(model=model, grid=True, connection=True, linewidth=0.1)
modelplot.show()

modelplot = ModelPlot(model=model, grid=False, connection=True, linewidth=0.1)
modelplot.show()

