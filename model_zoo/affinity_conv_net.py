def model(input_shape, output_shape):

    if input_shape[1] == 41:
        filter_size = 12
    else:
        filter_size = 10
    print(filter_size)

    # create model
    layer1 = {'layer': 'input', #41
            'input_shape': input_shape
            }
    layer2 = {'layer': 'conv1d',
            'num_filters': 24,
            'filter_size': filter_size,
            'norm': 'batch',
            'activation': 'leaky_relu',
            'dropout': 0.1,
            'padding': 'VALID',
            'mean_pool': 10,
            }
    layer3 = {'layer': 'conv1d',
            'num_filters': 96,
            'filter_size': 3,
            'norm': 'batch',
            'activation': 'leaky_relu',
            'dropout': 0.4,
            'padding': 'VALID',
            }
    layer4 = {'layer': 'dense',
            'num_units': output_shape[1],
            'activation': 'linear'
            }

    #from tfomics import build_network
    model_layers = [layer1, layer2, layer3, layer4]

    # optimization parameters
    optimization = {"objective": "squared_error",
                  "optimizer": "adam",
                  "learning_rate": 0.001,
                  "l2": 1e-6,
                  #"l1": 0.1,
                  }
    return model_layers, optimization
