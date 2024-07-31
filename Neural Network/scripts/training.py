import tensorflow as tf
import numpy as np
from scripts.models import Model
import csv
import matplotlib.pyplot as plt
import wandb
from scripts.utils import custom_metric, Wandb_plot, csv_logger, animate_function
import pandas as pd
import matplotlib.animation as animation
import ast
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter


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
config = wandb.config


x_train = np.arange(-10, 10, 0.01)
y_train = np.sin(x_train)
y_mean = np.mean(y_train)

loss_metric = custom_metric(name = "loss")
accuracy_metric = custom_metric(name = "accuracy")

metrics = {'loss': loss_metric, 'accuracy': accuracy_metric}

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate= 0.01)
m.compile(optimizer, metrics, y_mean)

w_callback = Wandb_plot()
logs = csv_logger(name = "test1.csv")
wandb.config.update({"model": m})

m.fit(x_train, y_train, epochs = 1500, batch_size = 128, callbacks = [w_callback, logs])
run.finish()

