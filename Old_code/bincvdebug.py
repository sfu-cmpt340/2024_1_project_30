# Creator: Ryan Vansickle
# Input: n > 2 classes of data
# Output: Trained DL model

import numpy as np
import keras
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt

image_size = (250, 250)
batch_size = 32

train, test = keras.utils.image_dataset_from_directory(
  'data/',
  labels='inferred',
  label_mode='int',
  color_mode='rgb',
  batch_size=batch_size,
  image_size=image_size,
  shuffle=True,
  seed=109,
  validation_split=0.2,
  subset='both',
  verbose=True
)

#TODO take advantage of colour channel redundancy?
def nine_sample():
  for images, labels in train.take(1):
    for i in range(0, min(len(labels), 9)):
      ax = plt.subplot(3, 3, i+1)
      plt.imshow(images[i].numpy()/255)
      plt.title("{}".format(labels[i].numpy()))
      plt.axis("off")
  plt.show()

preprocessing_layers =[
  # keras_cv.layers.Grayscale(output_channels=1) # grayscale
  # layers.RandomRotation(0.1), # random rotation
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
  #TODO figure out what a residual is
  
  for size in [256, 512, 728]:
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
  x = layers.Activation('relu')(x)
  
  x = layers.GlobalAveragePooling2D()(x)
  
  if num_classes == 2:
    units = 1
  else:
    units = num_classes
  
  x = layers.Dropout(0.25)(x)
  # We specify activation=None so as to return logits
  outputs = layers.Dense(units, activation=None)(x)
  return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
# keras.utils.plot_model(model, show_shapes=True)
# nine_sample()

epochs = 25

callbacks = [
  keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]

model.compile(
  optimizer=keras.optimizers.Adam(3e-4),
  loss=keras.losses.BinaryCrossentropy(from_logits=True),
  metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)

model.fit(
  train,
  epochs=epochs,
  callbacks=callbacks,
  validation_data=test
)

def testImage(model, path, showim=False):
  img = keras.utils.load_img(path, target_size=image_size)
  if showim == True:
    plt.imshow()
  img_array = keras.utils.img_to_array(img)
  img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis
  
  predictions = model.predict(img_array)
  score = float(keras.ops.sigmoid(predictions[0][0]))
  print("Image "+path+" score: ",score)

from random import randint

TEST_CASES_EACH = 5

for i in range(0, TEST_CASES_EACH):
  path = '0/NGS_' + str(randint(1, 29424)).zfill(5) + '.jpg'
  #print(path)
  testImage(model, path)

for i in range(0, TEST_CASES_EACH):
  path = '1/EBO_' + str(randint(1, 27395)).zfill(5) + '.jpg'
  #print(path)
  testImage(model, path)