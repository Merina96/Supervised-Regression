Supervised Regression Using Deep Learning

Project Overview

This project addresses a supervised regression problem using deep learning techniques.
The objective is to predict the (x, y) coordinates of a single pixel with value 255 in a 50×50 grayscale image, where all other pixels have a value of 0.

Each image contains exactly one white pixel placed at a random location, and the model is trained to learn the mapping between the image and the corresponding pixel coordinates.

Problem Statement

Given:

A 50×50 grayscale image

All pixel values are 0 except one pixel with value 255

The position of the white pixel is randomly assigned

Task:

Predict the (x, y) coordinates of the pixel with value 255 using deep learning

This problem is formulated as a supervised regression task, where the input is an image and the output is a pair of continuous values representing pixel coordinates.

Approach
Dataset Generation

A synthetic dataset is generated programmatically.

Each image is initialized as a 50×50 array of zeros.

One pixel is randomly selected and assigned a value of 255.

The corresponding (x, y) coordinates are stored as ground truth labels.

Synthetic data is appropriate because:

The problem is well-defined and controlled.

Exact ground truth labels are available.

Large datasets can be generated efficiently without manual annotation.

Model Architecture

Each image is flattened into a vector of 2500 values.

A fully connected neural network (Dense layers) is used.

The output layer contains two neurons corresponding to the predicted (x, y) coordinates.

Mean Squared Error (MSE) is used as the loss function.

A dense network is used as a baseline model to demonstrate learning behavior, as the assignment prioritizes approach and understanding over accuracy.

Training

The model is trained using the Adam optimizer.

A validation split is used to monitor generalization.

Training and validation loss are recorded during training.

Results and Observations

Training loss decreases significantly, indicating the model fits the training data.

Validation loss remains relatively high, suggesting limited generalization.

Predicted coordinates tend to cluster around the mean position.

This behavior is expected because flattening the image removes spatial information, and dense networks are not ideal for spatial localization tasks.

Visualizations Included

Training vs. Validation Loss graph

Scatter plot comparing actual and predicted (x, y) coordinates

These visualizations help evaluate model behavior and overfitting.

Dependencies

All required libraries are listed in requirements.txt.

Main dependencies include:

NumPy

Matplotlib

TensorFlow / Keras

Jupyter Notebook

How to Run the Project

Install dependencies:

pip install -r requirements.txt


Open the notebook:

jupyter notebook model.ipynb


Run all cells sequentially to generate the dataset, train the model, and visualize results.

Notes

The primary focus of this project is on problem formulation, dataset design, and learning behavior.

Accuracy is not the primary evaluation criterion.

A potential improvement is to use Convolutional Neural Networks (CNNs) to better capture spatial relationships.


Author
Merina Susan Cherian
ML Engineer