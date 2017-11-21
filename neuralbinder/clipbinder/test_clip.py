#---------------------------------------------------------------------------------------
"""
Summary: Test deep learning models on eclip test dataset
"""
#---------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
from six.moves import cPickle
import tensorflow as tf

from neuralbinder.neuralbindhelpers import helper
from neuralbinder.deepomics import neuralnetwork as nn
from neuralbinder.deepomics import utils, fit

#---------------------------------------------------------------------------------------------------

models = ['clip_conv_net', 'clip_residualbind']
ss_types = ['seq', 'pu']
window = 200

# training parameters
batch_size = 100
num_epochs = 200

# dataset path
dataset_path = '/media/peter/storage/encode_eclip/eclip_datasets'

# set results path
results_path = helper.make_directory('../../results', 'encode_eclip2')

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
        results = []
        for file_name in file_names:

            # parse filename
            index = file_name.index('.')
            rbp_name, cell_name, _ = file_name[:index].split('_')

            # load dataset
            dataset_file_path = os.path.join(dataset_path, file_name)
            train, valid, test = helper.load_dataset_hdf5(dataset_file_path, ss_type=ss_type)

            # build model
            input_shape = list(train['inputs'].shape)
            input_shape[0] = None
            output_shape = [None, train['targets'].shape[1]]

            # import model
            genome_model = helper.import_model(model)
            model_layers, optimization = genome_model(input_shape, output_shape)

            # build neural network class
            nnmodel = nn.NeuralNet(seed=247)
            nnmodel.build_layers(model_layers, optimization)
            #nnmodel.inspect_layers()

            # compile neural trainer
            model_save_path = os.path.join(sstype_path, rbp_name+'_'+cell_name)
            nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=model_save_path)

            # initialize session
            sess = utils.initialize_session(nnmodel.placeholders)

            # load best model
            nntrainer.set_best_parameters(sess)

            # test model on validation set
            loss, mean_vals, std_vals = nntrainer.test_model(sess, test, batch_size=128, name='test', verbose=1)

            # store results
            results.append(mean_vals)

            sess.close()
            # save results

        # store results
        with open(os.path.join(sstype_path, 'test_scores.pickle'), 'wb') as f:
            cPickle.dump(np.array(results), f, protocol=cPickle.HIGHEST_PROTOCOL)
