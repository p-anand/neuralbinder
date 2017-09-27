#---------------------------------------------------------------------------------------
"""
Summary: This script tests deep learning models on RNAcompete_2013 datasets with
sequence + secondary structure profiles, i.e. paired-unparied (pu) or structural
profiles (struct).
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
from deepomics import utils

#---------------------------------------------------------------------------------------

# different deep learning models to try out
old_models = ['conv_net', 'deep_residual_model', 'all_conv_net']
new_models = ['affinity_conv_net', 'affinity_residual_net', 'affinity_all_conv_net']
ss_types = ['struct'] # ['seq', 'pu', 'struct']
normalize_method = 'log_norm' #'log_norm'   # 'clip_norm'

data_path = '../../data/RNAcompete_2013/rnacompete2013.h5'
old_results_path = os.path.join('../../results', 'RNAcompete_2013_original')
results_path = os.path.join('../../results', 'RNAcompete_2013')

#---------------------------------------------------------------------------------------

# get list of rnacompete experiments
experiments = helper.get_experiments_hdf5(data_path)

# loop over different secondary structure contexts
for ss_type in ss_types:
	print('input data: ' + ss_type)
	sstype_path = helper.make_directory(results_path, normalize_method+'_'+ss_type)

	old_save_path = os.path.join(old_results_path, normalize_method+'_'+ss_type)

	# loop over different models
	for i, model in enumerate(old_models):
		print('model: ' + model)

		model_path = helper.make_directory(sstype_path, new_models[i])

		# loop over different RNA binding proteins
		for rbp_index in range(len(experiments)):

			try:
				experiment = experiments[rbp_index]
				print('Analyzing: '+ experiment)
				tf.reset_default_graph() # reset any tensorflow graphs
				np.random.seed(247) # for reproducibilitjy
				tf.set_random_seed(247) # for reproducibility

				# load rbp dataset
				train, valid, test = helper.load_dataset_hdf5(data_path, ss_type=ss_type, rbp_index=rbp_index)

				# process rbp dataset
				train, valid, test = helper.process_data(train, valid, test, method=normalize_method)
				print(train['inputs'].shape)

				# get shapes
				input_shape = list(train['inputs'].shape)
				input_shape[0] = None
				output_shape = train['targets'].shape

				# load model
				if model == 'all_conv_net':
					from models_old import all_conv_net as genome_model
				elif model == 'deep_residual_model':
					from models_old import deep_residual_model as genome_model
				elif model == 'conv_net':
					from models_old import conv_net as genome_model
				model_layers, optimization = genome_model.model(input_shape, output_shape)

				# build neural network class
				nnmodel = nn.NeuralNet(seed=247)
				nnmodel.build_layers(model_layers, optimization, use_scope=False)

				# compile neural trainer
				file_path = os.path.join(old_save_path, str(rbp_index)+'_'+model+'_'+ss_type)
				nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

				# initialize session
				sess = utils.initialize_session(nnmodel.placeholders)

				# test model
				nntrainer.set_best_parameters(sess)

				# test model
				#loss, mean_vals, std_vals = nntrainer.test_model(sess, test, batch_size=128, name='test', verbose=1)

				# save model parameters to new file path
				new_file_path = os.path.join(model_path, experiment+'_best.ckpt')
				nnmodel.save_model_parameters(sess, file_path=new_file_path)

				sess.close()
				
			except:
				with open('redo2.txt', 'a') as f:
					f.write(str(rbp_index)+' '+model+' '+ss_type+'\n')
