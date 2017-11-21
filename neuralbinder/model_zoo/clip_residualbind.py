def model(input_shape, output_shape):

    # create model
    layer1 = {'layer': 'input', #200
            'input_shape': input_shape
            }
    layer2 = {'layer': 'conv1d',
            'num_filters': 32,
            'filter_size': 13, # 200
            'norm': 'batch',
            'activation': 'relu',
            'dropout': 0.2,
            'padding': 'SAME',
            }
    layer3 = {'layer': 'conv1d_residual',
            'filter_size': 5,
            'function': 'relu',
            'dropout_block': 0.1,
            'dropout': 0.2,
            'mean_pool': 5, #40
            }
    layer4 = {'layer': 'conv1d',
            'num_filters': 96,
            'filter_size': 11, # 30
            'norm': 'batch',
            'activation': 'relu',
            'dropout': 0.2,
            'padding': 'VALID',
            }
    layer5 = {'layer': 'conv1d_residual',
            'filter_size': 5,
            'function': 'relu',
            'dropout_block': 0.1,
            'dropout': 0.4,
            'mean_pool': 10, # 3
            }
    layer6 = {'layer': 'conv1d',
            'num_filters': 192,
            'filter_size': 3, # 1
            'norm': 'batch',
            'activation': 'relu',
            'dropout': 0.5,
            'padding': 'VALID',
            }
    layer7 = {'layer': 'dense',
            'num_units': output_shape[1],
            'activation': 'sigmoid'
            }

    model_layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7]

    # optimization parameters
    optimization = {"objective": "binary",
                  "optimizer": "adam",
                  "learning_rate": 0.0001,
                  "l2": 1e-5,
                  #"label_smoothing": 0.05,
                  #"l1": 1e-6,
                  }
    return model_layers, optimization
