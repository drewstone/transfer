import numpy as np
import metrics
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt



def plot_acc(history, name):
    plt.clf()
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig("./plots/{}-accuracy.png".format(name))
    plt.clf()

    # summarize history for loss
    plt.clf()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig("./plots/{}-loss.png".format(name))
    plt.clf()

def plot_times(time_cb, name):
    plt.clf()
    plt.plot(time_cb[0].times, label="times")
    plt.title("{} model training times each epoch".format(name))
    plt.ylabel("times")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig("./plots/{}-times.png".format(name))
    plt.clf()

def plot_metrics(name, model, intermediate, shallow, x_test, y_test):
    main_result = metrics.matthews_correlation(model, x_test, y_test)
    interm_preds = intermediate.predict(x_test)
    shallow_result = metrics.matthews_correlation(shallow, interm_preds, y_test)

    y_pred, main_accuracies, best_threshold = main_result

    plt.clf()
    plt.figure(figsize=(40,9))
    plt.bar(np.arange(1,104), main_accuracies, align="center", alpha=0.5)
    plt.xticks(np.arange(1,104))
    plt.xlabel("document labels")
    plt.ylabel("accuracy")
    plt.title("main model accuracies using matthews coefficient")
    plt.savefig("./plots/{}-main-accuracies.png".format(name))
    plt.clf()

    y_pred, shallow_accuracies, best_threshold = shallow_result

    plt.figure(figsize=(40,9))
    plt.bar(np.arange(1,104), shallow_accuracies, align="center", alpha=0.5)
    plt.xticks(np.arange(1,104))
    plt.xlabel("document labels")
    plt.ylabel("accuracy")
    plt.title("shallow model accuracies using matthews coefficient")
    plt.savefig("./plots/{}-shallow-accuracies.png".format(name))
    plt.clf()

    differences = np.array(main_accuracies) - np.array(shallow_accuracies)

    plt.figure(figsize=(40,9))
    plt.bar(np.arange(1,104), differences, align="center", alpha=0.5)
    plt.xticks(np.arange(1,104))
    plt.xlabel("document labels")
    plt.ylabel("accuracy difference")
    plt.title("accuracy differences from main to shallow using matthews coefficient")
    plt.savefig("./plots/{}-difference-accuracies.png".format(name))
    plt.clf()

    return main_result, shallow_result