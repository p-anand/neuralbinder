#---------------------------------------------------------------------------------------
"""
Summary: Plot saliency for individual models for top-scoring
sequences in the setAB-mixed RNAcompete 2013 dataset.
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

from neuralbinder.neuralbindhelpers import helper, plot_helper
from neuralbinder.deepomics import neuralnetwork as nn
from neuralbinder.deepomics import utils, fit

#---------------------------------------------------------------------------------------

valid_frac = 0.1
test_frac = 0.25
num_epochs = 100
batch_size = 100

# different deep learning models to try out
models = ['affinity_conv_net', 'affinity_residualbind', 'affinity_all_conv_net']
normalize_method = 'log_norm'   # 'clip_norm'
ss_types = ['seq', 'pu']


data_path = neuralbinder.get_datasets('rnacompete2009.h5')
trained_path = '../../results/RNAcompete_2009'
results_path = utils.make_directory(trained_path, 'mixed_saliency')

#---------------------------------------------------------------------------------------

# get list of rnacompete experiments
rbp_names = ['Fusip', 'HuR', 'PTB', 'RBM4', 'SF2', 'SLM2', 'U1A', 'VTS1', 'YB1']

# loop over different secondary structure contexts
for ss_type in ss_types:
	print('input data: ' + ss_type)
  	sstype_path = helper.make_directory(trained_path, 'mixed_'+normalize_method+'_'+ss_type)

	# directory to save parameters of saliency
	saliency_sstype_path = helper.make_directory(results_path, 'mixed_'+normalize_method+'_'+ss_type)

	# loop over different models
	for model in models:
		print('model: ' + model)
		model_path = helper.make_directory(sstype_path, model)

		# sub-directory to save parameters of saliency
		saliency_model_path = helper.make_directory(saliency_sstype_path, model)

		# loop over different RNA binding proteins
		for rbp_name in rbp_names:
			print('Analyzing: '+ rbp_name)
			tf.reset_default_graph() # reset any tensorflow graphs
			np.random.seed(247) # for reproducibilitjy
			tf.set_random_seed(247) # for reproducibility

			# path to save saliency plots for each rbp experiment
			rbp_path = helper.make_directory(saliency_model_path, rbp_name)

			# load rbp dataset
			train, valid, test = helper.load_dataset_hdf5(data_path, dataset_name=rbp_name, ss_type=ss_type)

			# process rbp dataset
			train, valid, test = helper.process_data(train, valid, test, method=normalize_method)

			# shuffle set A and set B and then separate into training and test sequences
			inputs = np.vstack([train['inputs'], valid['inputs'], test['inputs']])
			targets = np.vstack([train['targets'], valid['targets'], test['targets']])
			num_data = targets.shape[0]
			shuffle = np.random.permutation(num_data)
			index = (np.cumsum([0, 1-valid_frac-test_frac, valid_frac, test_frac])*num_data).astype(int)
			train = {'inputs': inputs[index[0]:index[1]], 'targets': targets[index[0]:index[1]]}
			valid = {'inputs': inputs[index[1]:index[2]], 'targets': targets[index[1]:index[2]]}
			test = {'inputs': inputs[index[2]:index[3]], 'targets': targets[index[2]:index[3]]}

			# get shapes
			input_shape = list(train['inputs'].shape)
			input_shape[0] = None
			output_shape = train['targets'].shape

			# load model
			genome_model = helper.import_model(model)
			model_layers, optimization = genome_model(input_shape, output_shape)

			# build neural network class
			nnmodel = nn.NeuralNet(seed=247)
			nnmodel.build_layers(model_layers, optimization)

			# compile neural trainer
			file_path = os.path.join(model_path, rbp_name)
			nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

			# initialize session
			sess = utils.initialize_session(nnmodel.placeholders)

			# load best parameters
			nntrainer.set_best_parameters(sess)
			predictions = nntrainer.get_activations(sess, test, layer='output')
			sess.close()

			# plot scatter plot of prediction and experiment
			fig = plt.figure()
			plot_helper.scatter_plot(predictions, test['targets'], offset=0.3, alpha=0.4)
			outfile = os.path.join(rbp_path, rbp_name+'_scatter_performance.pdf')
			fig.savefig(outfile, format='pdf', dpi=50, bbox_inches='tight', rasterized=True)
			plt.close()

			# parameters for saliency analysis
			params = {'genome_model': genome_model,
					  'input_shape': input_shape,
					  'output_shape': output_shape,
					  'model_path': file_path+'_best.ckpt',
					  'optimization': optimization
					 }

			# plot top percentile saliency
			top_indices = plot_helper.get_percentile_indices(predictions[:,0], test['targets'][:,0], 99.9, 100.)
			plot_helper.plot_saliency_group(test, predictions, top_indices, params, rbp_path,
										name=rbp_name+'_top', ss_type=ss_type, num_plots=10)

			middle_indices = plot_helper.get_percentile_indices(predictions[:,0], test['targets'][:,0], 98, 98.5)
			plot_helper.plot_saliency_group(test, predictions, middle_indices, params, rbp_path,
										name=rbp_name+'_high', ss_type=ss_type, num_plots=5)

			low_indices = plot_helper.get_percentile_indices(predictions[:,0], test['targets'][:,0], 85, 90)
			plot_helper.plot_saliency_group(test, predictions, low_indices, params, rbp_path,
										name=rbp_name+'_middle', ss_type=ss_type, num_plots=5)


			# plot scatter plot of prediction and experiment
			fig = plt.figure()
			plot_helper.scatter_plot(predictions, test['targets'], offset=0.3, alpha=0.4)
			plt.scatter(predictions[top_indices,0], test['targets'][top_indices,0])
			plt.scatter(predictions[middle_indices,0], test['targets'][middle_indices,0])
			plt.scatter(predictions[low_indices,0], test['targets'][low_indices,0])
			outfile = os.path.join(rbp_path, rbp_name+'_scatter_selection.pdf')
			fig.savefig(outfile, format='pdf', dpi=50, bbox_inches='tight', rasterized=True)
			plt.close()

			# save results
			with open(os.path.join(rbp_path, 'scatter_coordinates.pickle'), 'wb') as f:
			    cPickle.dump(predictions, f, protocol=cPickle.HIGHEST_PROTOCOL)
			    cPickle.dump(test['targets'], f, protocol=cPickle.HIGHEST_PROTOCOL)
