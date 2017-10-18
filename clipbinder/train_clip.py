#---------------------------------------------------------------------------------------
"""
Summary: Train deep learning models on eclip datasets with
sequence and sequence with paired-unparied (pu) secondary structure profiles
"""
#---------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
import tensorflow as tf

sys.path.append('..')
import helper
from deepomics import neuralnetwork as nn
from deepomics import utils, fit

#---------------------------------------------------------------------------------------------------

models = ['clip_all_conv_net']#'clip_conv_net', 'clip_residual_net']
ss_types = ['seq', 'pu']
window = 200

# training parameters
batch_size = 100
num_epochs = 200

# dataset path
dataset_path = '/media/peter/storage/encode_eclip/eclip_datasets'

# set results path
results_path = helper.make_directory('../../results', 'encode_eclip')

# get list of .h5 files in dataset path
file_names = helper.get_file_names(dataset_path)

# loop through models
for model in models:

    # model results path
    model_path = helper.make_directory(results_path, model)

    # loop through secondary structure types
    for ss_type in ss_types:

        # model results path
        sstype_path = helper.make_directory(model_path, ss_type)

        # loop through each eclip dataset
        for file_name in file_names:

            # parse filename
            index = file_name.index('.')
            rbp_name, cell_name, _ = file_name[:index].split('_')

            # load dataset
            dataset_file_path = os.path.join(dataset_path, file_name)
            train, valid, test = helper.load_dataset_hdf5(dataset_file_path, ss_type=ss_type)

            # set shapes
            input_shape = list(train['inputs'].shape)
            input_shape[0] = None
            output_shape = [None, train['targets'].shape[1]]

            # import model
            genome_model = helper.import_model(model)
            model_layers, optimization = genome_model.model(input_shape, output_shape)

            # build neural network class
            nnmodel = nn.NeuralNet(seed=247)
            nnmodel.build_layers(model_layers, optimization)
            nnmodel.inspect_layers()

            # compile neural trainer
            model_save_path = os.path.join(sstype_path, rbp_name+'_'+cell_name)
            nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=model_save_path)

            # initialize session
            sess = utils.initialize_session(nnmodel.placeholders)

            # train model
            data = {'train': train, 'valid': valid}
            fit.train_minibatch(sess, nntrainer, data, batch_size=batch_size, num_epochs=num_epochs,
                                patience=25, verbose=2, shuffle=True, save_all=False)

            sess.close()
