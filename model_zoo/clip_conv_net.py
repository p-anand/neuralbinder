def model(input_shape, output_shape):

    # create model
    layer1 = {'layer': 'input', #200
            'input_shape': input_shape
            }
    layer2 = {'layer': 'conv1d',
            'num_filters': 24,
            'filter_size': 19, # 200
            'norm': 'batch',
            'activation': 'relu',
            'dropout': 0.1,
            'padding': 'SAME',
            'mean_pool': 50,  # 4
            }
    layer3 = {'layer': 'conv1d',
            'num_filters': 96,
            'filter_size': 4,
            'norm': 'batch',
            'activation': 'relu',
            'dropout': 0.4,
            'padding': 'VALID',
            }
    layer4 = {'layer': 'dense',
            'num_units': output_shape[1],
            'activation': 'sigmoid'
            }

    model_layers = [layer1, layer2, layer3, layer4]

    # optimization parameters
    optimization = {"objective": "binary",
                  "optimizer": "adam",
                  "learning_rate": 0.001,
                  "l2": 1e-6,
                  #"label_smoothing": 0.05,
                  #"l1": 1e-6,
                  }
    return model_layers, optimization
