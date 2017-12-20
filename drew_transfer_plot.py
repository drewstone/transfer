import os
import json
import argparse
import numpy as np
import transfer
import networks

nets = ["dnn", "mlp"]
layer_pairs = [(4,2), (4,4), (6,3), (6,6)]
neurons = [256, 512, 768, 1536, 2048]
amount=50000
interm_fraction=0.5

X, firstY, secondY = transfer.get_data('coarse')
# Split everything in half to make sure we aren't using overlapping data
X1, X2, Y1, Y2 = X[:int(X.shape[0]/2)], X[int(X.shape[0]/2):], firstY[:int(X.shape[0]/2)], secondY[int(X.shape[0]/2):]

# Find indices with only zeros to remove them in first data split
zero_indices = []
for inx, elt in enumerate(Y1):
    if elt.sum() < 1:
        zero_indices.append(inx)

indices = np.delete(np.arange(X1.shape[0]), zero_indices)
X1 = X1[indices]
Y1 = Y1[indices]

holdout = (X2[amount:amount*2].todense(), Y2[amount:amount*2])

# for file in os.listdir("./models"):
#     if "first" in file:
#         args = file.split("-")
#         network = args[2]
#         first_layers = int(float(args[4]))
#         second_layers = int(float(args[5]))
#         neuron_count = int(float(args[6].split("neurons")[0]))

#         first_model, intermediate, second_model, latent_model = networks.create(
#             network=network,
#             first_output_dim=3,
#             second_output_dim=1,
#             input_dim=X.shape[1],
#             first_layer_count=first_layers,
#             second_layer_count=second_layers,
#             interm_fraction=interm_fraction,
#             neuron_count=neuron_count)

#         first_model.load_weights("./models/{}".format(file), by_name=True)
#         transfer.save_intermediate(first_model, intermediate, name=file)

for net in nets:
    for pair in layer_pairs:
        for n_count in neurons:
            interm = "./models/intermediate-{}-{}-coarse-{}-{}-{}neurons.h5".format(amount, net, pair[0], pair[1], n_count)
            second = "./models/second-{}-{}-coarse-{}-{}-{}neurons.h5".format(amount, net, pair[0], pair[1], n_count)
            latent = "./models/latent-{}-{}-coarse-{}-{}-{}neurons.h5".format(amount, net, pair[0], pair[1], n_count)

            if (os.path.isfile(interm) and os.path.isfile(second) and os.path.isfile(latent)):
                first_model, intermediate, second_model, latent_model = networks.create(
                    network=net,
                    first_output_dim=3,
                    second_output_dim=1,
                    input_dim=X.shape[1],
                    first_layer_count=pair[0],
                    second_layer_count=pair[1],
                    interm_fraction=interm_fraction,
                    neuron_count=n_count)

                intermediate.load_weights(interm, by_name=True)
                latent_model.load_weights(latent, by_name=True)
                second_model.load_weights(second, by_name=True)

                print(interm)
                print(transfer.validate_holdout(second_model, holdout), second)
                print(transfer.validate_holdout(latent_model, holdout, intermediate), latent)


