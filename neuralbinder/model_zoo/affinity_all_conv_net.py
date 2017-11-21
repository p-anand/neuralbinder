def model(input_shape, output_shape):

    # create model
    layer1 = {'layer': 'input', #41
            'input_shape': input_shape
            }
    layer2 = {'layer': 'conv1d',
            'num_filters': 16,
            'filter_size': input_shape[1]-30,  # 32
            'strides': 1,  # 16
            'padding': 'SAME',
            'norm': 'batch',
            'activation': 'leaky_relu',
            }
    layer3 = {'layer': 'conv1d',
            'num_filters': 16,
            'filter_size': 10,  # 32
            'strides': 2,  # 16
            'padding': 'VALID',
            'norm': 'batch',
            'activation': 'leaky_relu',
            'dropout': 0.1,
            }
    layer4 = {'layer': 'conv1d',
            'num_filters': 32,
            'filter_size': 7,  # 16
            'strides': 1, # 5
            'padding': 'SAME',
            'norm': 'batch',
            'activation': 'leaky_relu',
            }
    layer5 = {'layer': 'conv1d',
            'num_filters': 32,
            'filter_size': 7,  # 10
            'norm': 'batch',
            'activation': 'leaky_relu',
            'dropout': 0.3,
            'strides': 2, # 5
            'padding': 'VALID',
            }
    layer6 = {'layer': 'conv1d',
            'num_filters': 64,
            'filter_size': 5,
            'norm': 'batch',
            'activation': 'leaky_relu',
            'dropout': 0.4,
            'padding': 'SAME',
            }
    layer7 = {'layer': 'conv1d',
            'num_filters': 64,
            'filter_size': 5,
            'norm': 'batch',
            'activation': 'leaky_relu',
            'dropout': 0.4,
            'padding': 'VALID',
            }
    layer8 = {'layer': 'dense',
            'num_units': output_shape[1],
            'activation': 'linear'
            }

    #from tfomics import build_network
    model_layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8]

    # optimization parameters
    optimization = {"objective": "squared_error",
                  "optimizer": "adam",
                  "learning_rate": 0.0005,
                  "l2": 1e-6,
                  #"l1": 0.1,
                  }
    return model_layers, optimization
