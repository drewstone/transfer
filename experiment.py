import numpy as np
import transfer
import networks
import plotting 

def run(network='dnn', amount=10009, val_split=0.67, split_type='simple'):

    # Fetch data and make simple split of data
    if split_type in ["random", "simple"]:
        X1, Y1, X2, Y2 = transfer.get_data(split_type=split_type, amt=amount)
    elif split_type in ["c_topics", "g_topics", "e_topics", "m_topics"]:
    	X1, Y1, X2, Y2, X3, Y3 = transfer.get_data(split_type=split_type, amt=amount)

    if network == "cnn":
        # Need to expand dimension for CNN to make sense
        X1, X2 = np.expand_dims(X1, axis=2), np.expand_dims(X2, axis=2)
        main, intermediate, shallow = networks.create_cnn()
    elif network == "dnn":
        main, intermediate, shallow = networks.create_dnn()
    elif network == 'mlp':
    	main, intermediate, shallow = networks.create_mlp()
    else:
        main, intermediate, shallow = networks.create_dnn()

    # Split data for training/testing for before and after transfer
    first_half = (X1, Y1) 
    second_half = (X2, Y2)

    # Train and transfer
    main, history, cbs = transfer.train_and_validate(main, data=first_half, validation_split=val_split)
    shallow, shallow_history, shallow_cbs = transfer.transfer_and_repeat(main, intermediate, shallow, data=second_half, validation_split=val_split)
    return (main, history, cbs), (shallow, shallow_history, shallow_cbs)
