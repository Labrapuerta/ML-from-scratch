import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import wandb
import plotly

class plot_training(tf.keras.callbacks.Callback):
    def __init__(self, save_images = False, epoch_interval = 1, wandb_ = False):
        super(plot_training, self).__init__()
        if not os.path.exists('Images'):
            os.makedirs('Images')
        self.path = os.getcwd()
        self.save_images = save_images
        self.epoch_interval = epoch_interval
        self.wandb_ = wandb_
    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        wandb.log({"loss": loss})
        if epoch % self.epoch_interval == 0:
            x = np.linspace(-2 * np.pi, 2 * np.pi, 100).reshape(-1, 1)
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            y = np.sin(x)
            y_hat = [self.model.predict(i)[0] for i in x]

'''            plt.figure(figsize=(17,5))
            plt.xlim(-2 * np.pi, 2 * np.pi)
            plt.ylim(-1, 1)
            plt.plot(x, y, label='True')
            plt.plot(x, y_hat, label='Model')
            plt.scatter(x, y_hat, color='red', s = 5)
            plt.text(-6,-0.80,f'Loss: {loss:.4f}', fontsize=12)
            for i in range(len(x)):
                plt.plot([x[i], x[i]], [y[i], y_hat[i]], color='black', linestyle='--', alpha=0.3)
            plt.legend()
            plt.title(f'Epoch {epoch}')
            if self.save_images:
                plt.savefig(f'{self.path}/Images/epoch_{epoch}.png')
            if self.wandb_:
                wandb.log({"chart": plt})
            plt.close()'''


class Wandb_plot(tf.keras.callbacks.Callback):
    def __init__(self):
        super(Wandb_plot, self).__init__()
    def on_epoch_end(self, epoch, logs={}):
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