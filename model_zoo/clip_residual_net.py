def model(input_shape, output_shape):

    # create model
    layer1 = {'layer': 'input', #100
            'input_shape': input_shape
            }
    layer2 = {'layer': 'conv1d',
            'num_filters': 64,
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
            'dropout': 0.3,
            'mean_pool': 10,
            }
    layer4 = {'layer': 'conv1d',
            'num_filters': 128,
            'filter_size': 5, # 20
            'norm': 'batch',
            'activation': 'relu',
            'dropout': 0.2,
            'padding': 'SAME',
            }
    layer5 = {'layer': 'conv1d_residual',
            'filter_size': 5,
            'function': 'relu',
            'dropout_block': 0.1,
            'dropout': 0.3,
            'max_pool': 5, # 4
            }
    layer6 = {'layer': 'conv1d',
            'num_filters': 256,
            'filter_size': 4, # 20
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
                  "learning_rate": 0.001,
                  "l2": 1e-6,
                  "label_smoothing": 0.05,
                  #"l1": 1e-6,
                  }
    return model_layers, optimization
