import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import keras

class PlotLosses(keras.callbacks.Callback):
    def __init__(self, name):
        self.name = name

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        plt.clf()
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.savefig('./plots/{}-loss-{}.png'.format(self.name, self.i));
        plt.clf()

class TimeHistory(keras.callbacks.Callback):
    def __init__(self, name):
        self.name = name

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def get_callbacks(name):
    timing = TimeHistory(name)
    plotlosses = PlotLosses(name)
    return [timing]