# FISHPainter

![logo](assets/FISHPainter.png)

## Overview

FISHPainter is a Python library designed for image processing and generation, specifically tailored for FISH (Fluorescence In Situ Hybridization) images. The library provides a range of functionalities, from preprocessing TIFF images to merging bounding boxes based on overlap criteria and an end-to-end workflow to generate a synthetic dataset based on user-specified criteria, including a library of DAPI stained images of fluorescence microscopy nuclei [1](https://zenodo.org/records/10798938).

## Features

- **Create Synthetic Dataset**: Functionality used to create the snythetic dataset using a user defined library for cell backgrounds, or the provided libary. ([create.py](https://github.com/SimonBon/FISHPainter/blob/main/FISHPainter/src/datasets/preprocess.py)). An overview of its usage is shown in the provided [Notebook](https://github.com/SimonBon/FISHPainter/blob/main/FISHPainter/notebook.ipynb)
- **Preprocessing**: Functions for reading TIFF images and normalizing them. ([preprocess.py](https://github.com/SimonBon/FISHPainter/blob/main/FISHPainter/src/preprocess.py))
- **Bounding Boxes**: Functions to determine if two bounding boxes should be merged and to merge bounding boxes for a given label. ([process_boxes.py](https://github.com/SimonBon/FISH-Painter/blob/main/FISHPainter/src/process_boxes.py))
- **Signal Generation**: Functions to create Gaussian signals and apply them to image patches. ([signals.py](https://github.com/SimonBon/FISHPainter/blob/main/FISHPainter/src/signals.py))
- **Utilities**: Functions to create datasets from bounding boxes. ([utils.py](https://github.com/SimonBon/FISHPainter/blob/main/FISHPainter/src/utils.py))

## Installation

To install the library you can use pip or clone the repository and install the required packages.

```bash
pip install FISHPainter
```
or

```bash
git clone https://github.com/SimonBon/FISHPainter.git
cd FISHPainter
pip install -r requirements.txt
```

## Creation of synthetic Dataset

```YAML

#definition of each condition and its parameters individually
CONDITION_0:                # Name is variable
#number of patches
  number: 100               # integer

  target_class: 0           # integer
#red defintion
  num_red: [2, 3]           # [low, high] or number
  num_red_cluster: [1, 2]   # low, high] or number
  red_cluster_size: [5, 10] # [low, high] or number

#green definition
  num_green: 2              # [low, high] or number
  num_green_cluster: 1      # [low, high] or number
  green_cluster_size: 6     # [low, high] or number
  
  signal_size: [1, 2]       #[low, high] or number


#definition of second condition
CONDITION_1:                # Name is variable
#number of patches
  number: 100               # integer

  target_class: 1           # integer
#red defintion
  num_red: 2                # [low, high] or number
  num_red_cluster: 0        # [low, high] or number
  red_cluster_size: 0       # [low, high] or number

#green definition
  num_green: [6, 10]        # [low, high] or number
  num_green_cluster: 0      # [low, high] or number
  green_cluster_size: 0     # [low, high] or number
  
#size definition
  signal_size: 1            # [low, high] or number
```
