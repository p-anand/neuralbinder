def model(input_shape, output_shape):

    # create model
    layer1 = {'layer': 'input', #41
            'input_shape': input_shape
            }
    layer2 = {'layer': 'conv1d',
            'num_filters': 32,
            'filter_size': 12,
            'norm': 'batch',
            'activation': 'relu',
            'dropout': 0.2,
            'padding': 'VALID',
            'max_pool': 10,
            }
    layer3 = {'layer': 'conv1d',
            'num_filters': 128,
            'filter_size': 3,
            'norm': 'batch',
            'activation': 'relu',
            'dropout': 0.5,
            'padding': 'VALID',
            }
    layer4 = {'layer': 'dense',
            'num_units': output_shape[1],
            'activation': 'linear'
            }

    model_layers = [layer1, layer2, layer3, layer4]

    # optimization parameters
    optimization = {"objective": "squared_error",
                  "optimizer": "adam",
                  "learning_rate": 0.001,
                  "l2": 1e-6,
                  #"l1": 1e-6,
                  }
    return model_layers, optimization
