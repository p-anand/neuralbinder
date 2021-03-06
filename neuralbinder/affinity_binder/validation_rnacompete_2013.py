#---------------------------------------------------------------------------------------
"""
Summary: Test deep learning models on RNAcompete_2013 validation datasets
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

#---------------------------------------------------------------------------------------

num_epochs = 100
batch_size = 100

# different deep learning models to try out
models = ['affinity_conv_net', 'affinity_residualbind', 'affinity_all_conv_net']
normalize_method = 'clip_norm'   # 'clip_norm' 'log_norm'
ss_types = ['seq']#, 'pu', 'struct']

data_path = neuralbinder.get_datasets('rnacompete2013.h5')
results_path = helper.make_directory('../../results', 'RNAcompete_2013')

#---------------------------------------------------------------------------------------

# get list of rnacompete experiments
experiments = helper.get_experiments_hdf5(data_path)

# loop over different secondary structure contexts
for ss_type in ss_types:
	print('input data: ' + ss_type)
	sstype_path = helper.make_directory(results_path, normalize_method+'_'+ss_type)

	# loop over different models
	for model in models:
		print('model: ' + model)
		model_path = helper.make_directory(sstype_path, model)

		# loop over different RNA binding proteins
		results = []
		for rbp_index, experiment in enumerate(experiments):
			print('Analyzing: '+ experiment)
			tf.reset_default_graph() # reset any tensorflow graphs
			np.random.seed(247) # for reproducibilitjy
			tf.set_random_seed(247) # for reproducibility

			# load rbp dataset
			train, valid, test = helper.load_dataset_hdf5(data_path, ss_type=ss_type, rbp_index=rbp_index)

			# process rbp dataset
			train, valid, test = helper.process_data(train, valid, test, method=normalize_method)

			# get shapes
			input_shape = list(train['inputs'].shape)
			input_shape[0] = None
			output_shape = train['targets'].shape

			# load model
			genome_model = helper.import_model(model)
			model_layers, optimization = genome_model(input_shape, output_shape)

			# build neural network class
			nnmodel = nn.NeuralNet(seed=247)
			nnmodel.build_layers(model_layers, optimization, use_scope=False)

			# compile neural trainer
			file_path = os.path.join(model_path, experiment)
			nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

			# initialize session
			sess = utils.initialize_session(nnmodel.placeholders)

			# load best model
			nntrainer.set_best_parameters(sess)

			# test model on validation set
			loss, mean_vals, std_vals = nntrainer.test_model(sess, valid, batch_size=128, name='valid', verbose=1)

			# store the Pearson correlation coefficient
			results.append(mean_vals[0])

			sess.close()

		# save results
		with open(os.path.join(model_path, 'validation_scores.pickle'), 'wb') as f:
			cPickle.dump(np.array(results), f, protocol=cPickle.HIGHEST_PROTOCOL)
