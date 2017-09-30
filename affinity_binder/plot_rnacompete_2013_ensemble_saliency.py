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

num_epochs = 100
batch_size = 100

# different deep learning models to try out
models = ['affinity_conv_net', 'affinity_residual_net', 'affinity_all_conv_net']
normalize_method = 'log_norm'   # 'clip_norm'
ss_types = ['seq', 'pu']

data_path = '../../data/RNAcompete_2013/rnacompete2013.h5'
trained_path = '../../results/RNAcompete_2013'
results_path = utils.make_directory(trained_path, 'saliency')

#---------------------------------------------------------------------------------------

# get list of rnacompete experiments
experiments = helper.get_experiments_hdf5(data_path)

# loop over different secondary structure contexts
for ss_type in ss_types:
	print('input data: ' + ss_type)
	sstype_path = os.path.join(trained_path, normalize_method+'_'+ss_type)

	# directory to save parameters of saliency
	saliency_sstype_path = helper.make_directory(results_path, normalize_method+'_'+ss_type)
	saliency_sstype_path = helper.make_directory(saliency_sstype_path, 'ensemble')

	# path to pre-trained model parameters
	best_path = os.path.join(trained_path, normalize_method+'_'+ss_type)

	# loop over different RNA binding proteins
	for rbp_index, experiment in enumerate(experiments):
		print('Analyzing: '+ experiment)
		tf.reset_default_graph() # reset any tensorflow graphs
		np.random.seed(247) # for reproducibilitjy
		tf.set_random_seed(247) # for reproducibility

		# path to save saliency plots for each rbp experiment
		rbp_path = helper.make_directory(saliency_sstype_path, experiment)
		print(rbp_path)

		# load rbp dataset
		train, valid, test = helper.load_dataset_hdf5(data_path, ss_type=ss_type, rbp_index=rbp_index)

		# process rbp dataset
		train, valid, test = helper.process_data(train, valid, test, method=normalize_method)

		input_shape = list(test['inputs'].shape)
		input_shape[0] = None
		output_shape = test['targets'].shape

		# get ensemble model predictions
		ensemble_predictions, predictions = helper.ensemble_predictions(test, experiment, models, input_shape, output_shape, best_path, use_scope=False)

		# plot scatter plot of prediction and experiment
		fig = plt.figure()
		plot_helper.scatter_plot(ensemble_predictions, test['targets'][:,0], offset=0.3, alpha=0.4)
		outfile = os.path.join(rbp_path, experiment+'_scatter_performance.pdf')
		fig.savefig(outfile, format='pdf', dpi=50, bbox_inches='tight', rasterized=False)
		plt.close()

		# plot top percentile saliency
		top_indices = plot_helper.get_percentile_indices(ensemble_predictions, test['targets'][:,0], 99.9, 100.)
		plot_helper.plot_ensemble_saliency_group(test, ensemble_predictions, top_indices, models, best_path, rbp_path,
												experiment, addon='top', ss_type=ss_type, num_plots=10, use_scope=False)

		middle_indices = plot_helper.get_percentile_indices(ensemble_predictions, test['targets'][:,0], 98, 98.5)
		plot_helper.plot_ensemble_saliency_group(test, ensemble_predictions, middle_indices, models, best_path, rbp_path,
												experiment, addon='high', ss_type=ss_type, num_plots=5, use_scope=False)

		low_indices = plot_helper.get_percentile_indices(ensemble_predictions, test['targets'][:,0], 85, 90)
		plot_helper.plot_ensemble_saliency_group(test, ensemble_predictions, low_indices, models, best_path, rbp_path,
												experiment, addon='middle', ss_type=ss_type, num_plots=5, use_scope=False)

		# plot scatter plot of prediction and experiment
		fig = plt.figure()
		plot_helper.scatter_plot(ensemble_predictions, test['targets'][:,0], offset=0.3, alpha=0.4)
		plt.scatter(ensemble_predictions[top_indices], test['targets'][top_indices,0])
		plt.scatter(ensemble_predictions[middle_indices], test['targets'][middle_indices,0])
		plt.scatter(ensemble_predictions[low_indices], test['targets'][low_indices,0])
		outfile = os.path.join(rbp_path, experiment+'_scatter_selection.pdf')
		fig.savefig(outfile, format='pdf', dpi=50, bbox_inches='tight', rasterized=True)
		plt.close()

		# save results
		with open(os.path.join(rbp_path, 'scatter_coordinates.pickle'), 'wb') as f:
		    cPickle.dump(ensemble_predictions, f, protocol=cPickle.HIGHEST_PROTOCOL)
		    cPickle.dump(test['targets'], f, protocol=cPickle.HIGHEST_PROTOCOL)
		    cPickle.dump(predictions, f, protocol=cPickle.HIGHEST_PROTOCOL)
