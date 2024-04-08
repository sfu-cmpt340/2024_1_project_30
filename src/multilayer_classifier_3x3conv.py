# Creator: Ryan Vansickle
# Input: n > 2 classes of cell data at PATH
# Output: trained model

# from google.colab import drive
# drive.mount('/content/drive')

import numpy as np
import keras
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt

# !unzip /content/drive/MyDrive/Datasets/BM_cytomorphology_data.zip -d /content/drive/MyDrive/Datasets/

image_size = (250, 250)
batch_size = 32

PATH = '/content/drive/MyDrive/Datasets/BM_cytomorphology_data/'

train, test = keras.utils.image_dataset_from_directory(
  PATH,
  labels='inferred',
  label_mode='categorical',
  color_mode='rgb',
  batch_size=batch_size,
  image_size=image_size,
  shuffle=True,
  seed=121,
  validation_split=0.2,
  subset='both',
  #verbose=True
)

num_classes = len(train.class_names)

preprocessing_layers =[
  # keras_cv.layers.Grayscale(output_channels=1) # grayscale
  layers.RandomRotation(0.1), # random rotation
  layers.Rescaling(1./255), # normalize (0,255) rgb range to (0,1)
  layers.RandomFlip(mode='horizontal_and_vertical') # random flip
]

def preprocess_data(images):
  for layer in preprocessing_layers:
    images = layer(images)
  return images
  
def make_model(input_shape, num_classes):
  inputs = keras.Input(shape=input_shape)
  x = preprocess_data(inputs)
  x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)

  previous_block_activation = x # Set aside residual

  for size in [256, 512, 1024]:
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Project residual
    residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)

    x = layers.add([x, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual

  x = layers.SeparableConv2D(1024, 3, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dense(128, activation="relu")(x)
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dropout(0.25)(x)

  outputs = layers.Dense(num_classes, activation="softmax")(x)

  model = keras.Model(inputs, outputs)

  return model

model = make_model(input_shape=image_size + (3,), num_classes=len(train.class_names))

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

epochs = 20
history = model.fit(train, validation_data=test, epochs=epochs, batch_size=128, verbose=1)

