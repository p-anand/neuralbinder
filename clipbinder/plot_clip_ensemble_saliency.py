#---------------------------------------------------------------------------------------
"""
Summary: This script trains deep learning models on RNAcompete_2013 datasets with
sequence + secondary structure profiles, i.e. paired-unparied (pu) or structural
profiles (struct).
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
import helper, plot_helper
from deepomics import neuralnetwork as nn
from deepomics import utils, fit

#---------------------------------------------------------------------------------------

models = ['clip_conv_net', 'clip_residual_net', 'clip_all_conv_net']
ss_types = ['seq', 'pu']
window = 200

# dataset path
dataset_path = '/media/peter/storage/encode_eclip/eclip_datasets'

# set results path
results_path = helper.make_directory('../../results', 'encode_eclip')
saliency_results_path = utils.make_directory(results_path, 'saliency')

# get list of .h5 files in dataset path
file_names = helper.get_file_names(dataset_path)

# loop through models
for model in models:
    # model results path
    model_path = os.path.join(results_path, model)

    saliency_model_path = helper.make_directory(saliency_results_path, model)

    # loop through secondary structure types
    for ss_type in ss_types:

        # model results path
        sstype_path = os.path.join(model_path, ss_type)
        saliency_sstype_path = helper.make_directory(saliency_model_path, ss_type)

		# loop through each eclip dataset
        results = []
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

            # get ensemble model predictions
            ensemble_predictions, predictions = helper.ensemble_clip_predictions(test, rbp_name+'_'+cell_name, models, input_shape, output_shape, sstype_path)

            # plot top percentile saliency
            plot_helper.plot_clip_ensemble_saliency_group(test, ensemble_predictions, top_indices, models, sstype_path, rbp_path,
            										rbp_name+'_'+cell_name, ss_type=ss_type, num_plots=20, use_scope=True)

            # save results
            with open(os.path.join(rbp_path, 'scatter_coordinates.pickle'), 'wb') as f:
                cPickle.dump(ensemble_predictions, f, protocol=cPickle.HIGHEST_PROTOCOL)
                cPickle.dump(test['targets'], f, protocol=cPickle.HIGHEST_PROTOCOL)
                cPickle.dump(predictions, f, protocol=cPickle.HIGHEST_PROTOCOL)
