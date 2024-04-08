# Creator: Nicola Murray
# Co-creator: Chau Pham
# Input: Databases from BM_cytomorphology_data and bone_marrow_cell_dataset
# Output: Confusion matrix, macro recall, precision, balanced and model accuracy 
# classification of cell type and accuracy score, keras model

# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, confusion_matrix, ConfusionMatrixDisplay
import os
from matplotlib.animation import FuncAnimation
from tensorflow.keras.utils import model_to_dot

from IPython.display import SVG
#NM Added the import of these libraries for metrics
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score, precision_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define the size of the images that we will feed into the network and the batch size for training
image_size = (250, 250)
batch_size = 32
data_path = 'Data/bone_marrow_cell_dataset/Test'

# Load the training dataset with a specific seed for reproducibility
train_ds = keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# Load the validation dataset in a similar way
val_ds = keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# Define the CNN model architecture function
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.RandomFlip("horizontal_and_vertical")(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model

# Determine the number of classes and create the model
num_classes = len(train_ds.class_names)
model = make_model(input_shape=image_size + (3,), num_classes=num_classes)

# Compile and train the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# Evaluate the model on the entire validation dataset
val_labels = []
val_predictions = []
for images, labels in val_ds:
    preds = model.predict(images)
    val_labels.extend(labels.numpy())
    val_predictions.extend(preds)

# Convert predictions and labels from one-hot to indices
val_labels_indices = np.argmax(val_labels, axis=1)
val_predictions_indices = np.argmax(val_predictions, axis=1)

# Calculate metrics for the entire validation set
macro_recall = recall_score(val_labels_indices, val_predictions_indices, average='macro')
print("Macro recall:", macro_recall)

precision = precision_score(val_labels_indices, val_predictions_indices, average='macro', zero_division=0)
print("Precision:", precision)

balanced_accuracy = balanced_accuracy_score(val_labels_indices, val_predictions_indices)
print("Balanced accuracy:", balanced_accuracy)

model_accuracy = accuracy_score(val_labels_indices, val_predictions_indices)
print("Model accuracy:", model_accuracy)

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(val_labels_indices, val_predictions_indices)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=train_ds.class_names)
disp.plot()
plt.show()


# Function to update the figure during the animation
def update_figure(frame_index):
    plt.clf()  # Clear the current figure
    image_file = image_files[frame_index]
    image_path = os.path.join(image_dir, image_file)
    img = keras.utils.load_img(image_path, target_size=image_size)
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = train_ds.class_names[predicted_class_index]
    confidence_score = predictions[0][predicted_class_index]

    # Show the image with the title including prediction and confidence
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class_label} | Confidence: {confidence_score:.2f}")
    plt.axis('off')  # Hide the axis


# Specify the directory where the images are located
image_dir = 'Data/bone_marrow_cell_dataset/Test/KSC/'

# List all files in the directory
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Sort the files to maintain a sequence
image_files.sort()

# Create a figure for plotting
fig = plt.figure()

# Create an animation, using the update_figure function, running through the images
ani = FuncAnimation(fig, update_figure, frames=len(image_files), repeat=False)

plt.show()

# Specify the directory where the images are located
image_dir = 'Data/bone_marrow_cell_dataset/Test/LYI/'

# List all files in the directory
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Sort the files to maintain a sequence
image_files.sort()

# Create a figure for plotting
fig = plt.figure()

# Create an animation, using the update_figure function, running through the images
ani = FuncAnimation(fig, update_figure, frames=len(image_files), repeat=False)

plt.show()

model.summary()



# Generate the dot format of a model
model_dot = model_to_dot(model, show_shapes=True, show_layer_names=True, dpi=200)

# If you are using IPython or Jupyter, you can display the SVG in the notebook.
SVG(model_dot.create(prog='dot', format='svg'))

# To save the dot file, you would use:
with open('Keras_Model.dot', 'w') as f:
    f.write(model_dot.to_string())

# Optionally, if you want to save the visualization as an image file directly
model_dot.write_png('Keras_Model.png')
# or for a PDF file
# model_dot.write_pdf('Keras_Model.pdf')
