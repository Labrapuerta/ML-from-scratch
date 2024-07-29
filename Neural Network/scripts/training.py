import tensorflow as tf
import numpy as np
from scripts.models import Model
import wandb
import csv
from scripts.utils import custom_metric, Wandb_plot



m = Model(1, [1,512,512,512,1], [None, tf.keras.activations.tanh,tf.keras.activations.tanh,tf.keras.activations.tanh,None])


wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="Neural Network from scratch", 
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "epochs": 1500,
        "batch_size": 128,
        "activation": "Tanh",
        "architecture" : "1,512,512,512,1",
        "Optimizer": "Adam",
        "Name" : "Tanh Model"
    },
)

