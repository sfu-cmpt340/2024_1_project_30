# SFU CMPT 340 Project Template -- Replace with project title
This repository is a template for your CMPT 340 course project.
Replace the title with your project title, and **add a snappy acronym that people remember (mnemonic)**.

(1-2 line summary) Take 1: Our project takes cell images from bone marrow smears as input and classifies them via feature extraction and training (ML). 
Take 2: Our project takes cell images obtained from bone marrow smears as input and trains with Machine Learning and Deep Learning to classify cells and extract their features. Then we output the results visually into models, plots and spreadsheets of values. 

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EcY0eyp7y3BNjzs8cBXSTZQBJapOrhw0k9x8XhcVyzLv0A?e=BIuzQI) | [Slack channel](https://sfucmpt340spring2024.slack.com/archives/C06DSE9RY3Y) | [Project report](https://www.overleaf.com/project/65a57eace7b3b293eb7481f8) |
|-----------|---------------|-------------------------|


- Timesheet: Link your timesheet (pinned in your project's Slack channel) where you track per student the time and tasks completed/participated for this project/
- Slack channel: Link your private Slack project channel.
- Project report: Link your Overleaf project report document.


## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.

[Video Link: currently a placeholder](https://www.youtube.com/watch?v=ChfEO8l-fas)


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)


<a name="demo"></a>
## 1. Example demo

As there are 5 separate .py programs to run, we will demo the work one by one.  

**DL_Cell_Identification:**

Epochs: 10
```bash
Found 107 files belonging to 2 classes.
Using 86 files for training.
Found 107 files belonging to 2 classes.
Using 21 files for validation.
Epoch 1/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 10s 3s/step - accuracy: 0.5039 - loss: 2.6392 - val_accuracy: 0.1905 - val_loss: 2.1128
Epoch 2/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 6s 2s/step - accuracy: 0.4240 - loss: 1.9267 - val_accuracy: 0.8095 - val_loss: 0.5096
Epoch 3/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 6s 2s/step - accuracy: 0.6539 - loss: 0.6284 - val_accuracy: 0.9524 - val_loss: 0.4899
Epoch 4/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 6s 2s/step - accuracy: 0.8269 - loss: 0.5221 - val_accuracy: 0.7619 - val_loss: 0.4202
Epoch 5/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 7s 3s/step - accuracy: 0.8483 - loss: 0.4140 - val_accuracy: 0.8095 - val_loss: 0.3005
Epoch 6/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 6s 2s/step - accuracy: 0.9241 - loss: 0.2903 - val_accuracy: 0.9048 - val_loss: 0.1942
Epoch 7/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 7s 2s/step - accuracy: 0.8852 - loss: 0.2948 - val_accuracy: 0.9048 - val_loss: 0.2827
Epoch 8/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 7s 2s/step - accuracy: 0.9144 - loss: 0.1935 - val_accuracy: 0.9048 - val_loss: 0.2514
Epoch 9/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 7s 2s/step - accuracy: 0.9106 - loss: 0.2299 - val_accuracy: 0.9048 - val_loss: 0.2682
Epoch 10/10
3/3 ━━━━━━━━━━━━━━━━━━━━ 8s 3s/step - accuracy: 0.9320 - loss: 0.1909 - val_accuracy: 0.9524 - val_loss: 0.1920
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)             │ (None, 250, 250, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ rescaling (Rescaling)                │ (None, 250, 250, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ random_flip (RandomFlip)             │ (None, 250, 250, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d (Conv2D)                      │ (None, 248, 248, 32)        │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 124, 124, 32)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 122, 122, 64)        │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 61, 61, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 59, 59, 128)         │          73,856 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_2 (MaxPooling2D)       │ (None, 29, 29, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 107648)              │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 128)                 │      13,779,072 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 2)                   │             258 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 41,617,736 (158.76 MB)
 Trainable params: 13,872,578 (52.92 MB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 27,745,158 (105.84 MB)
```
![ksc_animation](https://github.com/sfu-cmpt340/2024_1_project_30/assets/71118130/821eaa55-855a-499b-9519-388a5af701e3)
![lyi_animation 5 31 35 PM](https://github.com/sfu-cmpt340/2024_1_project_30/assets/71118130/761df35e-7e17-4de5-8ea6-2ae688d64803)
![Model Loss of KSC and LYI](https://github.com/sfu-cmpt340/2024_1_project_30/assets/71118130/e91d4c3b-5eae-4a83-95db-8359a91d3dc5)

 

**DL_Confusion_Matrix:**

- Macro recall: 0.9705882352941176
- Precision: 0.9
- Balanced accuracy: 0.9705882352941176
- Model accuracy: 0.9523809523809523

**Confusion Matrix of 21 Combined Training Images of KSC and LYI**
![Confusion Matrix LYI KSCpng](https://github.com/sfu-cmpt340/2024_1_project_30/assets/71118130/c3184fe8-c830-42d7-b3f9-b6d64b13b908)

**bincvdebug.py**

**Q3.py:**

**non_deep_ML.py:**

### Included in the Repo

```bash
repository
├── Old_code                     ## Storage of discarded or duplicate program files
├── Results                      ## Files and images outputted by the package
├── src                          ## source code of the package itself
├── README.md                    ## summary and project description for usage
```

<a name="installation"></a>

## 2. Installation

Before usage of this project, please make sure the following are installed: 

1. Python 3.11+
2. OpenCV (any version)
3. numpy, keras, tensorflow, matplotlib.pyplot, matplotlib.animation, neuralplot, pydot,graphviz, scikit-learn

Our program utilises tensorflow and neuralplot to produce Keras models of the provided databases during training while OpenCV and matlab plots are necessary for image processing and visualization. To install the libraries, paste the following commands into your IDE command window: 

```python
pip install neuralplot
pip install --upgrade keras-cv
pip install --upgrade keras-nlp
pip install --upgrade keras

# Needed for metrics
pip install -U scikit-learn

# tensorflow requires the latest version of pip
pip install --upgrade pip

# Anything above 2.10 is not supported on the GPU on Windows Native, macOS is fine though
pip install "tensorflow<2.11"

# For macOS:
brew install graphviz
# For Windows:
pip install graphviz

pip install os-sys
pip install matplotlib
```

Alternatively, all parts of our project can be run on Jupyter notebook or Google Colaboratory without the installation of additional imports except neuralplot. If any libraries or imports are missing, please follow the above python commands to install them. 

```bash
git clone https://github.com/sfu-cmpt340/2024_1_project_30.git
cd 2024_1_project_30

```

<a name="repro"></a>
## 3. Reproduction

For mounting google drive to colab for opening the dataset files:
```bash
from google.colab import drive
drive.mount('/content/drive')

#if drive.mount('/content/drive') doesn't work:
from google.colab import drive
drive.mount('/gdrive')

#for unzipping the datasets onto colab
!unzip path_to_file.zip -d path_to_directory
```
Make sure you approve ALL permissions for colab to access your google drive.
We recommend uploading the unzipped database files because unzipping on colab may take some time.

To run DL_Cell_Identification.py, DL_Confusion_Matrix.py and (Insert Ryan's code file name), you'll want the NGS and EBO datasets/or any two cell type classes from the database (2.6GB)
set it up in a folder like:
```bash
├── bincvdebug.py
├── DL_Cell_Identification.py
├── DL_Confusion_Matrix.py
├── data
│   ├── 0
│       ├── NGS_00001.jpg
│       ├── NGS_00002.jpg
│       └── NGS...
│   ├── 1
│       ├── EBO_00001.jpg
│       ├── EBO_00002.jpg
│       └── EBO...
```
For VSCode, you can run the code by pressing the button above in the right corner
![image](https://github.com/sfu-cmpt340/2024_1_project_30/assets/71118130/31b27fb1-3c2f-4fa8-8f73-75a30a32071e)


To run (Insert Evan's code file name), you'll want the (database insert here) and set it up in a folder like:

Important Reminders: 
- DL_Cell_Identification and DL_Confusion_Matrix: Currently, running the animations for the cell identification visualization only functions in VSCode and will not show up in Google Colaboratory or Jupyter Notebook.
- At this time, the Blood_Cancer dataset is non-compatible with the Deep Learning programs as the dataset uses tiff image files. The Deep Learning programs accept jpeg, jpg, png and such file types as input.
  
Datasets used in our analysis can be found: 
1. [Blood Cancer Cells Taken from Blood Smears](https://www.kaggle.com/datasets/akhiljethwa/blood-cancer-image-dataset)
2. [Annotated Dataset of Bone Marrow Cytology](https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/)
3. To be Added

Output will be saved in the following ways:
- The Keras model outputted by DL_Cell_Identification.py will be saved as PNG and DOT files.
- The confusion matrix generated by DL_Confusion_Matrix.py can be saved to a local drive as a PNG file. 
- The loss graph and cell identification can be saved to a local drive as a PNG file.


