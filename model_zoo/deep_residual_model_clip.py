def model(input_shape, output_shape):
    # create model
    layer1 = {'layer': 'input', #41
            'input_shape': input_shape
            }
    layer2 = {'layer': 'conv1d', 
            'num_filters': 96, 
            'filter_size': 12, 
            'norm': 'batch',
            'activation': 'relu',
            'dropout': 0.2,
            'padding': 'VALID',
            }
    layer3 = {'layer': 'conv1d_residual', 
            'filter_size': 5, 
            'function': 'relu',
            'dropout_block': 0.1,
            'dropout': 0.3,
            'mean_pool': 10,   
            }
    layer4 = {'layer': 'conv1d', 
            'num_filters': 192, 
            'filter_size': 3, 
            'norm': 'batch',
            'activation': 'relu',
            'dropout': 0.5,
            'padding': 'VALID',
            }
    layer5 = {'layer': 'dense', 
            'num_units': output_shape[1],
            'activation': 'sigmoid'
            }

    #from tfomics import build_network
    model_layers = [layer1, layer2, layer3, layer4, layer5]
    
    # optimization parameters
    optimization = {"objective": "binary",
                  "optimizer": "adam",
                  "learning_rate": 0.001, 
                  "l2": 1e-6,
                  #"l1": 0.1, 
                  }
    return model_layers, optimization
