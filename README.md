# SFU CMPT 340 Project Template -- Replace with project title
This repository is a template for your CMPT 340 course project.
Replace the title with your project title, and **add a snappy acronym that people remember (mnemonic)**.

(1-2 line summary) Our project takes cell images from bone marrow smears as input and classifies them via feature extraction and training (ML). 

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

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

A minimal example to showcase your work.
As there are 3/4 separate .py programs to run, we will demo the work one by one.  

```python
from amazing import amazingexample
imgs = amazingexample.demo()
for img in imgs:
    view(img)
```

### What to find where

Explain briefly what files are found where

```bash
repository
├── src                          ## source code of the package itself
├── scripts                      ## scripts, if needed
├── docs                         ## If needed, documentation   
├── README.md                    ## You are here
├── requirements.yml             ## If you use conda
```

<a name="installation"></a>

## 2. Installation

Provide sufficient instructions to reproduce and install your project. 
Provide _exact_ versions, test on CSIL or reference workstations.

Before usage of this project, please make sure the following are installed: 

1. Python 3.11+
2. OpenCV (any version)
3. numpy, keras, tensorflow, matplotlib.pyplot, matplotlib.animation, neuralplot, pydot,graphviz


Our program utilises tensorflow and neuralplot to produce Keras models of the NN during training while OpenCV and matlab plots are necessary for image processing and visualization.

Alternatively, all parts of our project can be ran on Jupyter notebook or Google Colaboratory without the installation of additional imports except neuralplot. To install neuralplot on Google Colaboratory, please type the following into the command window:

```python
!pip install neuralplot
pip install --upgrade keras-cv
pip install --upgrade keras-nlp
pip install --upgrade keras

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

```bash
git clone $THISREPO
cd $THISREPO
conda env create -f requirements.yml
conda activate amazing
```

<a name="repro"></a>
## 3. Reproduction
Demonstrate how your work can be reproduced, e.g. the results in your report.
```bash
mkdir tmp && cd tmp
wget https://yourstorageisourbusiness.com/dataset.zip
unzip dataset.zip
conda activate amazing
python evaluate.py --epochs=10 --data=/in/put/dir
```

For mounting google drive to colab for opening the dataset files:
```bash
from google.colab import drive
drive.mount('/content/drive') 
```

To run Ryan's and Chau's code, you'll want the NGS and EBO datasets (2.6GB)
set it up in a folder like:
```bash
├── bincvdebug.py
├── data
│   ├── 0
│       ├── NGS_00001.jpg
│       ├── NGS_00002.jpg
│       └── NGS...
│   ├── 1
│       ├── EBO_00001.jpg
│       ├── EBO_00002.jpg
│       └── EBO...```

Datasets used in our analysis can be found: 
1. [Blood Cancer Cells Taken from Blood Smears](https://www.kaggle.com/datasets/akhiljethwa/blood-cancer-image-dataset)
2. [Annotated Dataset of Bone Marrow Cytology](https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/)
3. To be Added

Output will be saved in ...

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 
