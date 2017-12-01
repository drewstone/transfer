import numpy as np
import metrics
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt



def plot_acc(name, first_history, second_history, latent_history):
    plt.clf()
    plt.plot(first_history.history["acc"], label="first train", color="r")
    plt.plot(first_history.history["val_acc"], label="first test", linestyle="--", color="r")
    plt.plot(second_history.history["acc"], label="second train", color="b")
    plt.plot(second_history.history["val_acc"], label="second test", linestyle="--", color="b")
    plt.plot(latent_history.history["acc"], label="latent train", color="c")
    plt.plot(latent_history.history["val_acc"], label="latent test", linestyle="--", color="c")
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
    plt.plot(latent_history.history["loss"], label="latent train", color="c")
    plt.plot(latent_history.history["val_loss"], label="latent test", linestyle="--", color="c")
    plt.title("model losses")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc="lower left")
    plt.savefig("./plots/{}-loss.png".format(name))
    plt.clf()

def plot_times(name, first_time_cb, second_time_cb, latent_time_cb):
    plt.clf()
    plt.plot(first_time_cb[0].times, label="first model times", color='r')
    plt.plot(second_time_cb[0].times, label="second model times", color='b')
    plt.plot(latent_time_cb[0].times, label="latent model times", color='c')
    plt.title("{} model training times each epoch".format(name))
    plt.ylabel("times")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig("./plots/{}-times.png".format(name))
    plt.clf()

def plot_metrics(name, intermediate, second_model, latent_model, x_test, y_test, indices, categories):
    second_acc, _ = metrics.matthews_correlation(second_model, x_test, y_test)

    interm_preds = intermediate.predict(x_test)
    latent_acc, _ = metrics.matthews_correlation(latent_model, interm_preds, y_test)

    w = 0.1

    cats = []
    for inx, cat in enumerate(categories.keys()):
        if inx in indices:
            cats.append(cat)

    plt.clf()
    plt.figure(figsize=(50,6))
    plt.bar(np.arange(1,y_test.shape[1]+1)-0.5*w, second_acc, align="center", alpha=0.75, color='r', label='second', width=0.1)
    plt.bar(np.arange(1,y_test.shape[1]+1)+0.5*w, latent_acc, align="center", alpha=0.75, color='b', label='latent', width=0.1)
    plt.xticks(np.arange(1,y_test.shape[1]+1), cats)
    plt.xlabel("document labels")
    plt.ylabel("accuracy")
    plt.title("model accuracies using matthews coefficient")
    plt.legend()
    plt.savefig("./plots/{}-main-accuracies.png".format(name))
    plt.clf()
