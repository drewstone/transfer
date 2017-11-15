import matplotlib as plt

def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from transfer import *
from networks import *
amount = 1000
val_split = 0.67
X1, Y1, X2, Y2 = [elt.todense() for elt in get_data(split_type='simple', amt=amount)]
main, intermediate, shallow = create_dnn()
first_half, second_half = (X1, Y1), (X2, Y2)
main, history = train_and_validate(main, data=first_half, validation_split=val_split)
shallow, shallow_history = transfer_and_repeat(main, intermediate, shallow, data=second_half, validation_split=val_split)