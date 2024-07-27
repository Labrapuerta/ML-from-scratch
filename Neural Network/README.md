# Neural Network to Approximate the Sine Function
Disclaimer, still under developement, but the clases are already implemented and the network is working, the only thing missing is the training loop and the results.

## Introduction
This project implements a neural network from scratch to approximate the sine function. Unlike typical neural network implementations, this project does not use pre-built neural network functions from libraries like TensorFlow or PyTorch.

## Table of Contents
- [Overview](#overview)
- [Implementation](#implementation)
- [Mathematics Behind Neurons](#mathematics-behind-neurons)
- [Results](#results)
- [Usage](#usage)
- [Jupyter Notebook](#jupyter-notebook)
- [References](#references)

## Overview

This project contains the following files:
- `NN_explanation.ipynb`: Explains the mathematics behind the neural network implementation, including the forward and backward propagation steps.
- `README.md`: This file.
- `Images/`: A folder containing images of the results.
- `scripts/`: A folder containing the python scripts for the project.

## Implementation Details
The neural network is implemented using the following classes:

#### Neuron: Represents a single neuron in the network.
Layer: Represents a layer consisting of multiple neurons.
Model: Represents the complete neural network model.

## Network Architecture
The network architecture consists of an input layer, hidden layers, and an output layer. Each layer contains neurons that apply weights, biases, and activation functions to the input data.

### Input Layer
The input layer directly takes the input data and passes it to the first hidden layer. Unlike other layers, the input layer does not apply any activation function, weights, or biases.

### Hidden Layers
Each hidden layer consists of multiple neurons that apply a specified activation function, for the project we will be comparing the performance of the ReLU, Sigmoid and Hyperbolic Tangent activation functions.

![Activation Functions](Images/Activation%20Functions.png)

### Output Layer
The output layer consists of a single neuron that provides the final output of the network, It will be using the linear activation function.

## Training the Network


## Results
The following images show the network's approximation of the sine function with the different activation functions compared to a baseline tf model.
You can see the comparison between the models on Wandb by clicking [here](https://wandb.ai/a01700257/Neural%20Network%20from%20scratch/table)


### ReLU Activation Function

### Sigmoid Activation Function

### Hyperbolic Tangent Activation Function


## Conclusion