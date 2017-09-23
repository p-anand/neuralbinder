from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

sys.path.append('..')
import helper
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency
np.random.seed(247)
tf.set_random_seed(247)


data_path = '../../data/RNAcompete_2013/rnacompete2013.h5'
experiments = helper.get_experiments_hdf5(data_path)


rbp_index = 69
ss_type = 'pu'
normalize_method = 'log_norm'

# load rbp dataset
train, valid, test = helper.load_dataset_hdf5(data_path, dataset_name=None, ss_type=ss_type, rbp_index=rbp_index)

# process rbp dataset
train, valid, test = helper.process_data(train, valid, test, method=normalize_method)
 
# set output file paths
from model_zoo import conv_net as genome_model

input_shape = list(train['inputs'].shape)
input_shape[0] = None
output_shape = train['targets'].shape
model_layers, optimization = genome_model.model(input_shape, output_shape)

# build neural network class
nnmodel = nn.NeuralNet(seed=247)
nnmodel.build_layers(model_layers, optimization)
nnmodel.inspect_layers()

# compile neural trainer
results_path = helper.make_directory('../../results', 'RNAcompete_2013')
save_path = helper.make_directory(results_path, normalize_method+'_'+ss_type)
file_path = os.path.join(save_path, str(rbp_index)+'_'+'conv_net'+'_'+ss_type)
nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

# initialize session
sess = utils.initialize_session(nnmodel.placeholders)

# fit data
data = {'train': train, 'valid': valid}
fit.train_minibatch(sess, nntrainer, data, batch_size=100, num_epochs=100, 
                      patience=20, verbose=2, shuffle=True, save_all=False)

# load best model
nntrainer.set_best_parameters(sess)

# test model
loss, mean_vals, std_vals = nntrainer.test_model(sess, test, batch_size=128, name='test', verbose=1)