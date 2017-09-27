#---------------------------------------------------------------------------------------
"""
Summary: This script trains deep learning models on RNAcompete_2013 datasets with
sequence + secondary structure profiles, i.e. paired-unparied (pu) or structural
profiles (struct).
"""
#---------------------------------------------------------------------------------------

from __future__ import print_function
import os, sys, h5py
import numpy as np
import tensorflow as tf

sys.path.append('..')
import helper
from deepomics import neuralnetwork as nn
from deepomics import utils

#---------------------------------------------------------------------------------------


data_path = '../../data/RNAcompete_2013/rnacompete2013.h5'
results_path = os.path.join('../../results', 'RNAcompete_2013_original')

# different deep learning models to try out
old_models = ['conv_net', 'deep_residual_model', 'all_conv_net']
new_models = ['affinity_conv_net', 'affinity_residual_net', 'affinity_all_conv_net']
ss_types = ['seq', 'pu', 'struct']

data_path = '../../data/RNAcompete_2013/rnacompete2013.h5'
old_results_path = os.path.join('../../results', 'RNAcompete_2013_original')
results_path = os.path.join('../../results', 'RNAcompete_2013')


# get list of rnacompete experiments
experiments = helper.get_experiments_hdf5(data_path)

rbp_index = 0
model = old_models[1]
normalize_method = 'log_norm'   # 'clip_norm'
ss_type = ss_types[0]

#---------------------------------------------------------------------------------------
print('Analyzing RBP='+str(rbp_index))
print(experiments[rbp_index])
print(model)
tf.reset_default_graph() # reset any tensorflow graphs
np.random.seed(247) # for reproducibilitjy
tf.set_random_seed(247) # for reproducibility


sstype_path = helper.make_directory(results_path, normalize_method+'_'+ss_type)

old_save_path = os.path.join(old_results_path, normalize_method+'_'+ss_type)

model_path = helper.make_directory(sstype_path, new_models[0])

# load rbp dataset
train, valid, test = helper.load_dataset_hdf5(data_path, ss_type=ss_type, rbp_index=rbp_index)

# process rbp dataset
train, valid, test = helper.process_data(train, valid, test, method=normalize_method)

# get shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None
output_shape = train['targets'].shape

# import model
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

# compile neural trainer
file_path = os.path.join(old_save_path, str(rbp_index)+'_'+model+'_'+ss_type)
#save_path = os.path.join(results_path, normalize_method+'_'+ss_type)
#file_path = os.path.join(save_path, str(rbp_index)+'_'+model+'_'+ss_type)
nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

# initialize session
sess = utils.initialize_session(nnmodel.placeholders)

# test model
nntrainer.set_best_parameters(sess)
nntrainer.test_model(sess, test)

sess.close()
