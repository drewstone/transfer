import numpy as np
import transfer
import networks
import plotting 

def run(network="dnn", amount=100, val_split=0.5, split_type="simple", name="experiment"):

    # Fetch data and make simple split of data
    X1, Y1, X2, Y2, first_ind, second_ind = transfer.get_data(split_type=split_type, amt=amount*2)

    # split data on specified axes
    for inx, elt in enumerate(Y1):
        if inx == 0:
            firstY1 = np.take(elt, first_ind)
        else:
            firstY1 = np.vstack((firstY1, np.take(elt, first_ind)))

    for inx, elt in enumerate(Y2):
        if inx == 0:
            secondY2 = np.take(elt, second_ind)
        else:
            secondY2 = np.vstack((secondY2, np.take(elt, second_ind)))

    if network == "dnn":
        first_model, intermediate, intermediate_transferred_model, second_model, shallow = networks.create_dnn(
            first_output_dim=len(first_ind),
            second_output_dim=len(second_ind),
            input_dim=X1.shape[1],
        )
    elif network == "mlp":
        first_model, intermediate, intermediate_transferred_model, second_model, shallow = networks.create_mlp(
            first_output_dim=len(first_ind),
            second_output_dim=len(second_ind),
            input_dim=X1.shape[1],
        )

    # Split data for training/testing for before and after transfer
    first_half = (X1[:amount], firstY1[:amount])
    second_half = (X2[:amount], secondY2[:amount])

    # Train main network on first data split
    train_result = transfer.train_and_validate(first_model, data=first_half, validation_split=val_split)
    trained_first_model, history, first_cb = train_result

    # Transfer weights and train shallow model on second data split
    transfer_result = transfer.transfer_and_repeat(trained_first_model, intermediate, shallow, data=second_half, validation_split=val_split)
    transferred_intermediate, trained_shallow, shallow_history, shallow_cb = transfer_result

    # Transfer weights to deep intermediate network and train on second data split
    intermediate_transferred_model = transfer.load_weights_by_name(trained_first_model, intermediate_transferred_model)
    train_result = transfer.train_and_validate(intermediate_transferred_model, data=second_half, validation_split=val_split)
    trained_interm_model, interm_history, interm_cb = train_result

    # Train last deep network on second data split
    train_result = transfer.train_and_validate(second_model, data=second_half, validation_split=val_split)
    trained_second_model, second_history, second_cb = train_result

    unique_name = '{}-{}-{}-{}'.format(amount, network, split_type, name)
    plotting.plot_acc(unique_name, history, interm_history, second_history, shallow_history)
    plotting.plot_times(unique_name, first_cb, interm_cb, second_cb, shallow_cb)
    plotting.plot_metrics(unique_name, transferred_intermediate, trained_shallow, trained_interm_model, trained_second_model, X2[amount:], secondY2[amount:])

    transfer.save_model(trained_first_model, "first-{}".format(unique_name))
    transfer.save_model(trained_interm_model, "interm-{}".format(unique_name))
    transfer.save_model(trained_second_model, "first-{}".format(unique_name))
    transfer.save_model(trained_shallow, "shallow-{}".format(unique_name))

if __name__ == '__main__':
    networks = ["dnn", "mlp"]
    split_types = ["simple", "random", "c_topics", "g_topics", "e_topics", "m_topics"]

    for inx, split_type in enumerate(split_types):
        for iinx, network in enumerate(networks):
            run(network=network, amount=40000, val_split=0.5, split_type=split_type, name="experiment")
