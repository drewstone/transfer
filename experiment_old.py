import numpy as np
import transfer
import networks
import plotting 

def run(network="dnn", amount=100, val_split=0.5, split_type="simple", name="experiment"):

    # Fetch data and make simple split of data
    X1, Y1, X2, Y2, X3, Y3 = transfer.get_data(split_type=split_type, amt=amount)

    if network == "cnn":
        # Need to expand dimension for CNN to make sense
        X1, X2, X3 = np.expand_dims(X1, axis=2), np.expand_dims(X2, axis=2), np.expand_dims(X3, axis=2)
        main, intermediate, shallow = networks.create_cnn()
        comparer, _, _ = networks.create_cnn()

    elif network == "dnn":
        main, intermediate, shallow = networks.create_dnn()
        comparer, _, _ = networks.create_dnn()
        
    elif network == "mlp":
        main, intermediate, shallow = networks.create_mlp()
        comparer, _, _ = networks.create_mlp()

    elif network == "wide_cnn":
        X1, X2, X3 = np.expand_dims(X1, axis=2), np.expand_dims(X2, axis=2), np.expand_dims(X3, axis=2)
        main, intermediate, shallow = networks.create_wider_cnn()
        comparer, _, _ = networks.create_wider_cnn()

    elif network == "large_cnn":
        X1, X2, X3 = np.expand_dims(X1, axis=2), np.expand_dims(X2, axis=2), np.expand_dims(X3, axis=2)
        main, intermediate, shallow = networks.create_large_cnn()
        comparer, _, _ = networks.create_large_cnn()

    else:
        main, intermediate, shallow = networks.create_dnn()

    # Split data for training/testing for before and after transfer
    first_half = (X1, Y1)
    second_half = (X2, Y2)

    # Train and transfer main and shallow networks
    main, history, cbs = transfer.train_and_validate(main, data=first_half, validation_split=val_split)
    intermediate, shallow, shallow_history, shallow_cbs = transfer.transfer_and_repeat(main, intermediate, shallow, data=second_half, validation_split=val_split)

    # Train comparer deep network to compare against shallow network
    comparer, comparer_history, comprarer_cbs = transfer.train_and_validate(comparer, data=second_half, validation_split=val_split)

    unique_name = "{}-{}-{}".format(network, split_type, name)
    transfer.save_model(main, "{}-main".format(unique_name))
    transfer.save_model(intermediate, "{}-intermediate".format(unique_name))
    transfer.save_model(shallow, "{}-shallow".format(unique_name))
    transfer.save_model(comparer, "{}-comparer".format(unique_name))

    # main_result, shallow_result = plotting.plot_metrics(unique_name, main, intermediate, shallow, X3, Y3)
    comparer_res, shallow_res = plotting.plot_metrics(unique_name, comparer, intermediate, shallow, X3, Y3)
    plotting.plot_acc(history, "main-{}".format(unique_name))
    plotting.plot_acc(shallow_history, "shallow-{}".format(unique_name))
    plotting.plot_acc(comparer_history, "comparer-{}".format(unique_name))

    return (main, history, cbs), (intermediate, shallow, shallow_history, shallow_cbs, shallow_res), (comparer, comparer_history, comprarer_cbs, comparer_res)

def run_new():
    val_split=0.5
    X1, Y1, X2, Y2, X3, Y3 = transfer.get_data(split_type='c_topics', amt=20000)

    main, intermediate, shallow = networks.create_dnn()
    seconddeep, _, _ = networks.create_dnn()

    first_half = (X1, Y1)
    second_half = (X2, Y2)

    main, history, cbs = transfer.train_and_validate(main, data=first_half, validation_split=val_split)
    intermediate, shallow, shallow_history, shallow_cbs = transfer.transfer_and_repeat(main, intermediate, shallow, data=second_half, validation_split=val_split)

    seconddeep, second_history, second_cbs = transfer.train_and_validate(seconddeep, data=second_half, validation_split=val_split)

    plotting.plot_metrics()

def run_dnns(amount=50000, val_split=0.5, name="exp"):
    networks = ["dnn", "mlp"]
    split_types = ["random", "c_topics", "g_topics", "e_topics", "m_topics"]

    for inx, split_type in enumerate(split_types):
        for iinx, network in enumerate(networks):
            run(network=network, amount=amount, val_split=val_split, split_type=split_type, name=name)

def run_cnns(amount=50000, val_split=0.5, name="cnn-exp"):
    networks = ["cnn", "wide_cnn"]
    split_types = ["random", "c_topics", "g_topics", "e_topics", "m_topics"]

    for inx, split_type in enumerate(split_types):
        for iinx, network in enumerate(networks):
            run(network=network, amount=amount, val_split=val_split, split_type=split_type, name=name)

if __name__ == "__main__":
    run(network='dnn', amount=50000, val_split=0.5, split_type='c_topics', name='test')
