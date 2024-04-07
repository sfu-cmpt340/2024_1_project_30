# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
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
data_path = 'Data/bone_marrow_cell_dataset/Test'  # Path where the training and validation datasets are located

# Load the training dataset with a specific seed for reproducibility
# The dataset is expected to be in `data_path`, split into subdirectories for each class
# The function automatically infers classes from subdirectories in `data_path`
# We use a validation split of 20%, so 80% of the data will be used for training
train_ds = keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'  # Labels will be one-hot encoded
)

# Load the validation dataset in a similar way
# Here, the remaining 20% of the data is used for validation
val_ds = keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'  # Labels will be one-hot encoded
)

# Define the CNN model architecture function
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)  # Input layer with the shape of our images
    x = layers.Rescaling(1.0 / 255)(inputs)  # Rescale pixel values from [0, 255] to [0, 1]
    x = layers.RandomFlip("horizontal_and_vertical")(x)  # Data augmentation: random flips
    x = layers.Conv2D(32, 3, activation="relu")(x)  # First convolutional layer
    x = layers.MaxPooling2D()(x)  # First pooling layer to reduce spatial dimensions
    x = layers.Conv2D(64, 3, activation="relu")(x)  # Second convolutional layer
    x = layers.MaxPooling2D()(x)  # Second pooling layer
    x = layers.Conv2D(128, 3, activation="relu")(x)  # Third convolutional layer
    x = layers.MaxPooling2D()(x)  # Third pooling layer
    x = layers.Flatten()(x)  # Flatten the 3D output to 1D for the dense layers
    x = layers.Dense(128, activation="relu")(x)  # First dense layer
    x = layers.Dropout(0.5)(x)  # Dropout layer to prevent overfitting
    outputs = layers.Dense(num_classes, activation="softmax")(x)  # Output layer with softmax activation for multiclass classification

    model = keras.Model(inputs, outputs)  # Create the model
    return model

# Determine the number of classes from the training dataset
num_classes = len(train_ds.class_names)
# Create the model with the defined input shape and number of classes
model = make_model(input_shape=image_size + (3,), num_classes=num_classes)
# Compile the model with Adam optimizer and categorical crossentropy loss for multiclass classification
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train the model with the training dataset and validate it with the validation dataset
epochs = 25 #shouldn't be more than 20 
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

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
    
    
    #NM added the following code for metrics
    #Record the true classes of each cell in the validation set and put into an array to compare with the predicted classes
    true_classes = np.concatenate([y for x, y in val_ds], axis=0)



    ## Sensitivity / Recall Metrics ##

    # Calclate the recall score for each class. In order of the classes provided in class_names
    classwise_recall = recall_score(true_classes, predicted_class_index, average=None)
    print("Class-specific recall = ", classwise_recall)

    # Calclate the recall score for each class then takes the average of them,
    macro_recall = recall_score(true_classes, predicted_classes_index, average='macro')
    print("Macro recall = ", macro_recall)

    # Compute the recall score for each individual prediction, treating them as equals regardless of class before taking the average.
    # Accounts for label imbalance. Currently commented out due to amount of time it would take to run. Reinstate as needed.
    #micro_recall = recall_score(true_classes, predicted_classes_index, average='micro')
    #print("Micro recall = ", micro_recall)


    ## Precision Metrics ##

    # Find the highest probability in each prediction.
    pred_score = np.max(predictions, axis=1)
    print("Prediction score = ", pred_score)


    # Calculate the average precision score for each class.
    precision = precision_score(true_classes, predicted_classes_index, labels = class_names, average = None)
    print("Precision = ", precision)

    # Calculate the precision of the entire validation set, accounting for label imbalance.
    # Currently commented out due to amount of time it would take to run. Reinstate as needed.
    #avg_global_precision = average_precision_score(true_classes, pred_score, average = 'micro')


    ## Accuracy metrics ##

    # Calculate the accuracy for the entire model that takes the number of cases in each class into account.
    # The classes in our dataset are inequal in size. This should account for that.
    bal_accuracy = balanced_accuracy_score(true_classes, predicted_classes_index)
    print("Balanced accuracy = ", bal_accuracy)

    model_accuracy = accuracy_score(true_classes, predicted_classes_index)
    print("Model  accuracy = ", model_accuracy)



    ## Confusion Matrix ##

    # Find the class names to be used in the confusion matrix - SHOULD THIS BE ABOVE CALL TO PREDICT?
    class_names = train_ds.class_names

    # Calculate and display the confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes_index, labels = class_names)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels = class_names)
    disp.plot()
    plt.show()


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
with open('model_architecture.dot', 'w') as f:
    f.write(model_dot.to_string())

# Optionally, if you want to save the visualization as an image file directly
model_dot.write_png('model_architecture.png')
# or for a PDF file
# model_dot.write_pdf('model_architecture.pdf')

# Plot the training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
