# Creator: Nicola Murray
# Co-creator: Chau Pham
# Input: Databases from BM_cytomorphology_data and bone_marrow_cell_dataset
# Output: Confusion matrix, macro recall, precision, balanced and model accuracy 
# classification of cell type and accuracy score, keras model

# Script to calcualte the metrics we will be using to evaluate our model.
# These metrics will also allow us to compare our model with models from papers.

#Import libraries needed for metrics
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score, precision_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# This code is meant to be incorporated into Chau's code to evaluate the model. The following 4 lines are from Chau.
# The metrics code will be inserted beneath into the function update figure:
#  predictions = model.predict(img_array)
#  predicted_class_index = np.argmax(predictions[0])
#  predicted_class_label = train_ds.class_names[predicted_class_index]
#  confidence_score = predictions[0][predicted_class_index]



# Find which class has the highest predicted value for each test image and put them into an array
# predicted_classes = np.argmax(predictions, axis=1)
# The equivalent to this variable in Chau's code is predicted_class_index, will be using that instead


# Record the true classes of each cell in the validation set and put into an array to compare with the predicted classes
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
