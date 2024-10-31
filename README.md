# README

## Overview

This repository contains a modified Fully Convolutional Network (FCN) designed for UAV (Unmanned Aerial Vehicle) imagery segmentation. The primary goal is to provide efficient segmentation on high-resolution UAV datasets by leveraging data augmentation and deep learning techniques. The FCN model architecture has been adapted from the standard FCN structure to fit specific requirements of UAV-based image datasets. 

> **Note:** The dataset used in this project is personal and restricted for public use due to privacy considerations. The original FCN structure and techniques have been modified to improve compatibility and performance for this dataset.

## Model Architecture

The model used in this project is based on a modified version of the Fully Convolutional Network (FCN) for segmentation. 

A variation with and without data augmentation has been implemented to test the robustness of the model against various transformations. 

## Dataset

The model was developed and tested using a UAV dataset, which includes high-resolution images and corresponding ground truth masks. Due to confidentiality constraints, the dataset is not available in this repository. The dataset is organized as follows:
- **Training Data:** `/images/train/images` and `/images/train/masks`
- **Validation Data:** `/images/val/images` and `/images/val/masks`

## Data Augmentation

Data augmentation is performed to improve model generalization. The augmentations include random rotations, shifts, zoom, horizontal flips, and rescaling. An option to disable augmentation is also available to evaluate model performance on unaltered images.

## Training

The model can be trained with the following setup:
- **Batch Size:** 8
- **Epochs:** 100
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Metrics:** Accuracy, Precision, F1 Score, Mean IoU

Callbacks for early stopping, model checkpoints, and custom visualizations are included to monitor training progress. A visualization callback saves prediction samples at the end of each epoch, aiding in qualitative analysis of model performance.

## Evaluation

The model performance is evaluated on the validation set using:
- **Precision**
- **F1 Score**
- **Mean IoU (Intersection over Union)**

Additionally, a confusion matrix and a set of training/validation loss and accuracy plots are generated.

## Results Visualization

Images showing model predictions, ground truth masks, and inputs are saved during training to assist in qualitative assessment. Inference time per image is calculated for further evaluation.

## Usage

To run the model, first set up the required directories and install necessary dependencies:
```bash
pip install -r requirements.txt
```

The main script can be executed to train and evaluate the model:
```bash
python train_fcn.py
```

### Key Files
- **model.py:** Contains the FCN model architecture.
- **train_fcn.py:** Main script for loading data, training, and evaluating the model.
- **visualization.py:** Handles the visualization of predictions and metrics.

## License

This project is licensed under the MIT License. Please see the `LICENSE` file for more details. 

## Acknowledgments

- This work is based on the Fully Convolutional Networks for Semantic Segmentation (Long et al., 2015). The paper can be found [here](https://arxiv.org/abs/1411.4038).
- The UAV dataset used is proprietary and restricted from public access.
