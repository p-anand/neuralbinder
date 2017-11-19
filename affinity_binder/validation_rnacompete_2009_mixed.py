#---------------------------------------------------------------------------------------
"""
Summary: Test deep learning models on setAB-mixed RNAcompete_2009 validation datasets
"""
#---------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
from six.moves import cPickle
import tensorflow as tf

sys.path.append('..')
import helper
from deepomics import neuralnetwork as nn
from deepomics import utils, fit

#---------------------------------------------------------------------------------------

valid_frac = 0.1
test_frac = 0.25
num_epochs = 100
batch_size = 100

# different deep learning models to try out
models = ['affinity_conv_net', 'affinity_residualbind', 'affinity_all_conv_net']
normalize_method = 'log_norm'   # 'clip_norm'
ss_types = ['seq', 'pu', 'struct']

data_path = '../../data/RNAcompete_2009/rnacompete2009.h5'
results_path = helper.make_directory('../../results', 'RNAcompete_2009')

#---------------------------------------------------------------------------------------

# get list of rnacompete experiments
rbp_names = ['Fusip', 'HuR', 'PTB', 'RBM4', 'SF2', 'SLM2', 'U1A', 'VTS1', 'YB1']

# loop over different secondary structure contexts
all_results = {}
for ss_type in ss_types:
	print('input data: ' + ss_type)
	sstype_path = helper.make_directory(results_path, 'mixed_'+normalize_method+'_'+ss_type)

	# loop over different models
	for model in models:
		print('model: ' + model)
		model_path = helper.make_directory(sstype_path, model)

		# loop over different RNA binding proteins
		results = []
		for rbp_name in rbp_names:
			print('Analyzing: '+ rbp_name)
			tf.reset_default_graph() # reset any tensorflow graphs
			np.random.seed(247) # for reproducibilitjy
			tf.set_random_seed(247) # for reproducibility

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
