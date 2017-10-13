def model(input_shape, output_shape):

    # create model
    layer1 = {'layer': 'input', #200
            'input_shape': input_shape
            }
    layer2 = {'layer': 'conv1d',
            'num_filters': 16,
            'filter_size': 19,  # 182
            'strides': 1,  # 16
            'padding': 'VALID',
            'norm': 'batch',
            'activation': 'leaky_relu',
            }
    layer3 = {'layer': 'conv1d',
            'num_filters': 16,
            'filter_size': 9,  # 174
            'strides': 2,  # 87
            'padding': 'VALID',
            'norm': 'batch',
            'activation': 'leaky_relu',
            'dropout': 0.1,
            }
    layer4 = {'layer': 'conv1d',
            'num_filters': 32,
            'filter_size': 8,  # 80
            'strides': 2, # 40
            'padding': 'VALID',
            'norm': 'batch',
            'dropout': 0.1,
            'activation': 'leaky_relu',
            }
    layer5 = {'layer': 'conv1d',
            'num_filters': 32,
            'filter_size': 8,  # 34
            'norm': 'batch',
            'activation': 'leaky_relu',
            'dropout': 0.3,
            'strides': 2, # 17
            'padding': 'VALID',
            'dropout': 0.2,
            }
    layer6 = {'layer': 'conv1d',
            'num_filters': 48,
            'filter_size': 8,  # 10
            'norm': 'batch',
            'activation': 'leaky_relu',
            'strides': 2, # 5
            'dropout': 0.3,
            'padding': 'VALID',
            }
    layer7 = {'layer': 'conv1d',
            'num_filters': 64,
            'filter_size': 5,
            'norm': 'batch',
            'activation': 'leaky_relu',
            'strides': 2, #
            'dropout': 0.4,
            'padding': 'VALID',
            }
    layer8 = {'layer': 'dense',
            'num_units': output_shape[1],
            'activation': 'sigmoid'
            }

    #from tfomics import build_network
    model_layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8]

    # optimization parameters
    optimization = {"objective": "binary",
                  "optimizer": "adam",
                  "learning_rate": 0.001,
                  "l2": 1e-6,
                  #"l1": 0.1,
                  }
    return model_layers, optimization
