import json
import argparse
import numpy as np
import transfer
import networks
import plotting 

# data_frac: fraction of the amount (data) that we want to train the first network on
def run(name, network, amount, data_frac, val_split, first_layers, second_layers, interm_fraction, neuron_count_per_layer):
    until = amount*2

    # Fetch data and make simple split of data
    X, firstY, secondY = transfer.get_data('coarse')

    # Split everything in half to make sure we aren't using overlapping data
    X1, X2, Y1, Y2 = X[:int(X.shape[0]*data_frac)], X[int(X.shape[0]*data_frac):], firstY[:int(X.shape[0]*data_frac)], secondY[int(X.shape[0]*data_frac):]

    # Find indices with only zeros to remove them in first data split
    zero_indices = []
    for inx, elt in enumerate(Y1):
        if elt.sum() < 1:
            zero_indices.append(inx)

    indices = np.delete(np.arange(X1.shape[0]), zero_indices)
    X1 = X1[indices]
    Y1 = Y1[indices]

    # Gather models
    first_model, intermediate, second_model, latent_model = networks.create(
        network=network,
        first_output_dim=3,
        second_output_dim=1,
        input_dim=X.shape[1],
        first_layer_count=first_layers,
        second_layer_count=second_layers,
        interm_fraction=interm_fraction,
        neuron_count=neuron_count_per_layer)

    print(first_model.summary(), intermediate.summary(), second_model.summary(), latent_model.summary())

    # Split data for training/testing for before and after transfer
    fst = (X1[:amount].todense(), Y1[:amount])
    snd = (X2[:amount].todense(), Y2[:amount])
    holdout = (X2[amount:until].todense(), Y2[amount:until])

    # Train main network on first data split
    first_model, first_history, first_cb = transfer.train_and_validate(
        epochs=10,
        model=first_model,
        data=fst,
        validation_split=val_split)

    # Train second deep network on second data split
    second_model, second_history, second_cb = transfer.train_and_validate(
        epochs=10,
        model=second_model,
        data=snd,
        validation_split=val_split)

    intermediate, latent_model, latent_history, latent_cb = transfer.transfer_and_repeat(
        epochs=10,
        model=first_model,
        intermediate=intermediate,
        transfer_model=latent_model,
        data=snd,
        validation_split=val_split)

    print(transfer.validate_holdout(second_model, holdout))
    print(transfer.validate_holdout(latent_model, holdout, intermediate))

    # plot and save models
    unique_name = '{}-{}-{}-{}'.format(amount, network, 'coarse', name)
    
    plotting.plot_acc(
        name=unique_name,
        first_history=first_history,
        second_history=second_history,
        latent_history=latent_history)

    transfer.save_model(first_model, "first-{}".format(unique_name))
    transfer.save_model(second_model, "second-{}".format(unique_name))
    transfer.save_model(latent_model, "latent-{}".format(unique_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--network')
    parser.add_argument('--amount', '-a')
    parser.add_argument('--data_frac')
    parser.add_argument('--first_layers')
    parser.add_argument('--second_layers')
    parser.add_argument('--fraction', '-f')
    parser.add_argument('--neurons', '-n')
    parser.add_argument('--name')

    args = parser.parse_args()

    print("********************** Running experiment **********************")
    print("Network: {}".format(args.network))
    print("Amount: {}".format(args.amount))
    print("First layers: {}".format(args.first_layers))
    print("Second layers: {}".format(args.second_layers))
    print("Intermediate fraction: {}".format(args.fraction))
    print("Neurons: {}".format(args.neurons))

    run(network=args.network,
        amount=int(float(args.amount)),
        data_frac=float(args.data_frac),
        val_split=0.5,
        name=args.name,
        first_layers=int(float(args.first_layers)),
        second_layers=int(float(args.second_layers)),
        interm_fraction=float(args.fraction),
        neuron_count_per_layer=int(float(args.neurons)))
            
