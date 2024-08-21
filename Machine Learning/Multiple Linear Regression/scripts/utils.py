import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import matplotlib.animation as animation
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter 

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
                writer.writerow(['Epoch', 'Accuracy', 'Loss','Beta0', 'Beta1'])
            writer.writerow([epoch, logs.get('Accuracy'), logs.get('Loss'), self.model.coeff.w[0].numpy()[0], self.model.coeff.w[1].numpy()[0]])
        file.close()

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

#### Creating a simple animation based on the logs    
def animate_function(logs, X, y, name = 'training_animation.gif'):
    df = pd.read_csv(logs)
    # Prepare the figure and axis
    fig, ax = plt.subplots(figsize=(17, 5))
    line, = ax.plot([], [], label='Model', lw=2, color='red')
    text = ax.text(8.5, 2, '', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    vertical_lines = []
    title = ax.text(0.5, 1.05, 'Epoch', transform=ax.transAxes, ha="center", fontsize=16)
    ax.legend(loc='upper right')

    # Set initial values
    def init():
        ax.set_xlim(0,10)
        ax.set_ylim(0, 35)
        ax.scatter(X, y, c = 'blue')
        return line, text

    # Update the plot for each frame
    def update(frame):
        epoch_data = df.iloc[frame]
        epoch = epoch_data['Epoch']
        loss = epoch_data['Loss']
        y_hat = X * epoch_data['Beta1'] + epoch_data['Beta0']
        line.set_data(X, y_hat)
        text.set_text(f'Loss: {loss:.4f} \nAccuracy: {epoch_data["Accuracy"]:.4f}')
        title.set_text(f'Epoch {epoch}')

        # Remove old vertical lines
        for vline in vertical_lines:
            vline.remove()
        vertical_lines.clear()
        
        # Add new vertical lines
        for i in range(len(X)):
            vline = ax.plot([X[i], X[i]], [y[i], y_hat[i]], color='black', linestyle='--', alpha=0.3)
            vertical_lines.append(vline[0])
        
        return line, text, *vertical_lines

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True)
    ani.save(f'Images/{name}', writer='imagemagick')
    plt.close()
