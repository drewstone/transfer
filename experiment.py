import json
import argparse
import numpy as np
import transfer
import networks
import plotting 

def run():
    name = 'exp'
    network = 'dnn'
    split_type = 'coarse'
    amount = 50000
    until = amount*2
    final = amount*3
    val_split=0.5

    # Fetch data and make simple split of data
    X, firstY, secondY = transfer.get_data('coarse')

    # Gather models
    first_model, intermediate, second_model, latent_model = networks.create(
        network=network,
        first_output_dim=3,
        second_output_dim=1,
        input_dim=X.shape[1],
        first_layer_count=4,
        second_layer_count=2,
        interm_fraction=0.5,
        neuron_count=32)

    print(first_model.summary(), intermediate.summary(), second_model.summary(), latent_model.summary())

    # Split data for training/testing for before and after transfer
    fst = (X[:amount].todense(), firstY[:amount])
    snd = (X[amount:until].todense(), secondY[amount:until])
    holdout = (X[until:final].todense(), secondY[until:final])

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
    unique_name = '{}-{}-{}-{}'.format(amount, network, split_type, name)
    
    plotting.plot_acc(
        name=unique_name,
        first_history=first_history,
        second_history=second_history,
        latent_history=latent_history)

    transfer.save_model(first_model, "first-{}".format(unique_name))
    transfer.save_model(second_model, "second-{}".format(unique_name))
    transfer.save_model(latent_model, "latent-{}".format(unique_name))

if __name__ == '__main__':
    run()
            
