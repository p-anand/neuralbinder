#---------------------------------------------------------------------------------------
"""
Summary:  Generate saliency plots for individual models as well as
the ensemble of models.
"""
#---------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle
import tensorflow as tf

sys.path.append('..')
import neralbinder.neuralbindhelpers.helper as helper
import neralbinder.neuralbindhelpers.plot_helper as plot_helper
from neuralbinder.deepomics import neuralnetwork as nn
from neuralbinder.deepomics import utils, fit

#---------------------------------------------------------------------------------------

models = ['clip_conv_net', 'clip_residualbind']
ss_types = ['seq', 'pu']
window = 200

# dataset path
dataset_path = '/media/peter/storage/encode_eclip/eclip_datasets'

# set results path
results_path = helper.make_directory('../../results', 'encode_eclip')
saliency_results_path = utils.make_directory(results_path, 'saliency_ensemble')

# get list of .h5 files in dataset path
file_names = helper.get_file_names(dataset_path)

# loop through models
for model in models:

    # model results path
    model_path = helper.make_directory(results_path, model)

    saliency_model_path = helper.make_directory(saliency_results_path, model)

    # loop through secondary structure types
    for ss_type in ss_types:

        # model results path
        sstype_path = helper.make_directory(model_path, ss_type)

        saliency_sstype_path = helper.make_directory(saliency_model_path, ss_type)

        # loop through each eclip dataset
        for file_name in file_names:

            # parse filename
            index = file_name.index('.')
            rbp_name, cell_name, _ = file_name[:index].split('_')
            rbp_path = helper.make_directory(saliency_sstype_path, rbp_name+'_'+cell_name)

            # load dataset
            dataset_file_path = os.path.join(dataset_path, file_name)
            train, valid, test = helper.load_dataset_hdf5(dataset_file_path, ss_type=ss_type)

            input_shape = list(test['inputs'].shape)
            input_shape[0] = None
            output_shape = test['targets'].shape

			# load model
            genome_model = helper.import_model(model)
            model_layers, optimization = genome_model(input_shape, output_shape)

			# build neural network class
	    nnmodel = nn.NeuralNet(seed=247)
            nnmodel.build_layers(model_layers, optimization)

			# compile neural trainer
            model_save_path = os.path.join(sstype_path, rbp_name+'_'+cell_name)
            nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=model_sae_path)

			# initialize session
            sess = utils.initialize_session(nnmodel.placeholders)

			# load best parameters
            nntrainer.set_best_parameters(sess)
	    predictions = nntrainer.get_activations(sess, test, layer='output')
	    sess.close()

			# parameters for saliency analysis
	    params = {'genome_model': genome_model,
		      'input_shape': input_shape,
                      'output_shape': output_shape,
                      'model_path': model_sae_path+'_best.ckpt',
                      'optimization': optimization
                     }

			# plot top percentile saliency
            top_indices = np.argmax(predictions)[::-1]
	    plot_helper.plot_saliency_group(test, predictions, top_indices, params, rbp_path, name=rbp_name+'_top', ss_type=ss_type, num_plots=10)
