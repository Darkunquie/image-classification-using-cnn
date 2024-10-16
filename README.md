# Image Classification with Convolutional Neural Networks

## Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images as either dogs or cats. The dataset consists of labeled images, and the model is trained to distinguish between the two classes.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

You can install the required packages using pip:


pip install tensorflow numpy matplotlib
##Dataset
The dataset consists of images stored in CSV format:

input.csv: Training images
labels.csv: Labels for training images (0 for dog, 1 for cat)
input_test.csv: Testing images
labels_test.csv: Labels for testing images
Ensure that the images are formatted as 100x100 pixels with 3 color channels (RGB).

##**Usage**
Load your data into the specified CSV files.
Run the provided code in a Jupyter Notebook or Google Colab.
Adjust the number of epochs and batch size as needed for your dataset.
Results
After training, the model's accuracy on the test set is displayed. A prediction for a random test image is also provided.

##Acknowledgments
TensorFlow and Keras for providing the deep learning framework.
NumPy for numerical operations.
Matplotlib for data visualization.
