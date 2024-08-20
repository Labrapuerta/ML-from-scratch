import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import wandb
import csv
import pandas as pd
import matplotlib.animation as animation
import ast
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter 


class Wandb_plot(tf.keras.callbacks.Callback):
    def __init__(self, epoch_plot = 0):
        super(Wandb_plot, self).__init__()
        self.epoch_plot = epoch_plot
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.epoch_plot == 0:
            x = np.arange(-10, 10, 0.1)
            y = np.sin(x)

            y_hat = [self.model.predict(i)[0] for i in x]

            wandb.log({"Model evolution" : wandb.plot.line_series(
                            xs = x.flatten(),
                            ys = [y.flatten(), np.array(y_hat).flatten()],
                            keys=["Sin(x)", "Model(x)"],
                            title=f"Model Evolution {epoch}",
                            xname="x")})
        wandb.log({"Loss": logs.get('loss')})
        wandb.log({"Accuracy": logs.get('accuracy')})

class csv_logger(tf.keras.callbacks.Callback):
    def __init__(self, path = 'logs', name = 'log.csv', **kwargs):
        super(csv_logger, self).__init__(**kwargs)
        self.path = f'{os.getcwd()}/{path}'
        self.name = name
        if os.path.isfile(f'{self.path}/{self.name}'):
            os.remove(f'{self.path}/{self.name}')
        self.x  = np.arange(-10, 10, 0.1)

    def on_epoch_end(self, epoch, logs={}):
        with open(f'{self.path}/{self.name}', mode='a') as file:
            writer = csv.writer(file)
            if epoch == 0:
                writer.writerow(['Epoch', 'Accuracy', 'Loss', 'Prediction'])
            y_hat = [self.model.predict(i).item() for i in self.x]
            writer.writerow([epoch, logs.get('accuracy'), logs.get('loss'), y_hat])

        file.close()
#### Custom Metric for training the model

class custom_metric(tf.keras.metrics.Metric):
    def __init__(self, name='custom_metric'):
        super(custom_metric, self).__init__(name= name)
        self.custom_metric = self.add_weight(name= name, initializer='zeros')
        self.count = self.add_weight(name= 'Count', initializer='zeros')

    def update_state(self, value):
        self.custom_metric.assign_add(value)
        self.count.assign_add(1.0)

    def result(self):
        return self.custom_metric / self.count

    def reset_state(self): 
        self.custom_metric.assign(0.0)
        self.count.assign(0.0)


def animate_function(logs, name = 'training_animation.gif'):
    df = pd.read_csv(logs)
    df.Prediction = df.Prediction.apply(lambda x: ast.literal_eval(x))
    x = np.arange(-10, 10, 0.1)
    y = np.sin(x)
    # Prepare the figure and axis
    fig, ax = plt.subplots(figsize=(17, 5))
    line, = ax.plot([], [], label='Model', lw=2, color='red')
    true_line, = ax.plot([], [], label='Sin(x)', lw=2, color='blue')
    text = ax.text(-6, -0.8, '', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    vertical_lines = []
    title = ax.text(0.5, 1.05, 'Epoch', transform=ax.transAxes, ha="center", fontsize=16)
    ax.legend(loc='upper right')

    # Set initial values
    def init():
        ax.set_xlim(-2 * np.pi, 2 * np.pi)
        ax.set_ylim(-1, 1)
        true_line.set_data(x, y)
        return line, true_line, text

    # Update the plot for each frame
    def update(frame):
        epoch_data = df.iloc[frame]
        epoch = epoch_data['Epoch']
        loss = epoch_data['Loss']
        y_hat = epoch_data['Prediction']
        
        line.set_data(x, y_hat)
        text.set_text(f'Loss: {loss:.4f}')
        title.set_text(f'Epoch {epoch}')

        # Remove old vertical lines
        for vline in vertical_lines:
            vline.remove()
        vertical_lines.clear()
        
        # Add new vertical lines
        for i in range(len(x)):
            vline = ax.plot([x[i], x[i]], [y[i], y_hat[i]], color='black', linestyle='--', alpha=0.3)
            vertical_lines.append(vline[0])
        
        return line, true_line, text, *vertical_lines

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True)
    ani.save(f'Images/{name}', writer='imagemagick')
    plt.close()