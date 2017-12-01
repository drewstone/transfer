import argparse
import numpy as np
import transfer
import networks
import plotting 

def run(network="dnn", amount=100, val_split=0.5, split_type="simple", name="experiment",
        layer_count=10, interm_fraction=0.25, neuron_count_per_layer=32):

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

    # Gather models
    first_model, intermediate, second_model, latent_model = networks.create(
        network=network,
        first_output_dim=len(first_ind),
        second_output_dim=len(second_ind),
        input_dim=X1.shape[1],
        layer_count=layer_count,
        interm_fraction=interm_fraction,
        neuron_count=neuron_count_per_layer
    )

    print(first_model.summary())
    print(second_model.summary())
    print(latent_model.summary())

    # Split data for training/testing for before and after transfer
    first_half = (X1[:amount], firstY1[:amount])
    second_half = (X2[:amount], secondY2[:amount])

    # Train main network on first data split
    first_model, first_history, first_cb = transfer.train_and_validate(
        model=first_model,
        data=first_half,
        validation_split=val_split
    )

    # Train second deep network on second data split
    second_model, second_history, second_cb = transfer.train_and_validate(
        model=second_model,
        data=second_half,
        validation_split=val_split
    )

    intermediate, latent_model, latent_history, latent_cb = transfer.transfer_and_repeat(
        model=first_model,
        intermediate=intermediate,
        transfer_model=latent_model,
        data=second_half,
        validation_split=val_split
    )

    # plot and save models
    unique_name = '{}-{}-{}-{}'.format(amount, network, split_type, name)
    
    plotting.plot_acc(
        name=unique_name,
        first_history=first_history,
        second_history=second_history,
        latent_history=latent_history,
    )


    plotting.plot_times(
        name=unique_name,
        first_time_cb=first_cb,
        second_time_cb=second_cb,
        latent_time_cb=latent_cb,
    )

    plotting.plot_metrics(
        name=unique_name,
        intermediate=intermediate,
        second_model=second_model,
        latent_model=latent_model,
        x_test=X2[amount:],
        y_test=secondY2[amount:]
    )

    transfer.save_model(first_model, "first-{}".format(unique_name))
    transfer.save_model(second_model, "second-{}".format(unique_name))
    transfer.save_model(latent_model, "latent-{}".format(unique_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--random', '-r', action='store_true')
    parser.add_argument('--simple', '-s', action='store_true')
    parser.add_argument('--ctopics', '-c', action='store_true')
    parser.add_argument('--gtopics', '-g', action='store_true')
    parser.add_argument('--etopics', '-e', action='store_true')
    parser.add_argument('--mtopics', '-m', action='store_true')
    parser.add_argument('--mlp', action='store_true')
    parser.add_argument('--dnn', action='store_true')
    parser.add_argument('--name', '-n')
    parser.add_argument('--amount', '-a')

    args = parser.parse_args()

    if args.dnn:
        network='dnn'
    elif args.mlp:
        network='mlp'

    if args.random:
        split = 'random'
    elif args.simple:
        split = 'simple'
    elif args.ctopics:
        split = 'c_topics'
    elif args.gtopics:
        split = 'g_topics'
    elif args.etopics:
        split = 'e_topics'
    elif args.mtopics:
        split = 'm_topics'

    run(network=network, amount=int(float(args.amount)), val_split=0.5, split_type=split, name=args.name)
            
