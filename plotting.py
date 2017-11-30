import numpy as np
import metrics
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt



def plot_acc(name, first_history, interm_history, second_history, shallow_history):
    plt.clf()
    plt.plot(first_history.history["acc"], label="first train", color="r")
    plt.plot(first_history.history["val_acc"], label="first test", linestyle="--", color="r")
    plt.plot(second_history.history["acc"], label="second train", color="b")
    plt.plot(second_history.history["val_acc"], label="second test", linestyle="--", color="b")
    plt.plot(interm_history.history["acc"], label="interm train", color="c")
    plt.plot(interm_history.history["val_acc"], label="interm test", linestyle="--", color="c")
    plt.plot(shallow_history.history["acc"], label="shallow train", color="y")
    plt.plot(shallow_history.history["val_acc"], label="shallow test", linestyle="--", color="y")
    plt.title("model accuracies")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc="lower right")
    plt.savefig("./plots/{}-accuracy.png".format(name))
    plt.clf()

    # summarize history for loss
    plt.clf()
    plt.plot(first_history.history["loss"], label="first train", color="r")
    plt.plot(first_history.history["val_loss"], label="first test", linestyle="--", color="r")
    plt.plot(second_history.history["loss"], label="second train", color="b")
    plt.plot(second_history.history["val_loss"], label="second test", linestyle="--", color="b")
    plt.plot(interm_history.history["loss"], label="interm train", color="c")
    plt.plot(interm_history.history["val_loss"], label="interm test", linestyle="--", color="c")
    plt.plot(shallow_history.history["loss"], label="shallow train", color="y")
    plt.plot(shallow_history.history["val_loss"], label="shallow test", linestyle="--", color="y")
    plt.title("model losses")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc="lower left")
    plt.savefig("./plots/{}-loss.png".format(name))
    plt.clf()

def plot_times(name, first_time_cb, interm_time_cb, second_time_cb, shallow_time_cb):
    plt.clf()
    plt.plot(first_time_cb[0].times, label="first model times", color='r')
    plt.plot(second_time_cb[0].times, label="second model times", color='b')
    plt.plot(interm_time_cb[0].times, label="interm model times", color='c')
    plt.plot(shallow_time_cb[0].times, label="shallow model times", color='y')
    plt.title("{} model training times each epoch".format(name))
    plt.ylabel("times")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig("./plots/{}-times.png".format(name))
    plt.clf()

def plot_metrics(name, intermediate, shallow, deep_interm, deep_second, x_test, y_test):
    interm_acc, _ = metrics.matthews_correlation(deep_interm, x_test, y_test)
    second_acc, _ = metrics.matthews_correlation(deep_second, x_test, y_test)

    interm_preds = intermediate.predict(x_test)
    shallow_acc, _ = metrics.matthews_correlation(shallow, interm_preds, y_test)

    w = 0.1

    plt.clf()
    plt.figure(figsize=(50,6))
    plt.bar(np.arange(1,y_test.shape[1]+1)-w, interm_acc, align="center", alpha=0.75, color='r', label='interm', width=0.1)
    plt.bar(np.arange(1,y_test.shape[1]+1), second_acc, align="center", alpha=0.75, color='b', label='second', width=0.1)
    plt.bar(np.arange(1,y_test.shape[1]+1)+w, shallow_acc, align="center", alpha=0.75, color='c', label='shallow', width=0.1)
    plt.xticks(np.arange(1,y_test.shape[1]+1))
    plt.xlabel("document labels")
    plt.ylabel("accuracy")
    plt.title("model accuracies using matthews coefficient")
    plt.legend()
    plt.savefig("./plots/{}-main-accuracies.png".format(name))
    plt.clf()
