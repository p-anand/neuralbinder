#---------------------------------------------------------------------------------------
"""
Summary: Train deep learning models on a single eCLIP dataset.
"""
#---------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
import tensorflow as tf

from neuralbinder.neuralbindhelpers import helper
from neuralbinder.deepomics import neuralnetwork as nn
from neuralbinder.deepomics import utils, fit

#----------------------------------------------------------------------------------------
#rbp_names = ['TIA1', 'RBFOX2', 'PTBP1', 'HNRNPC', 'TIA1', 'RBFOX2', 'PTBP1', 'PUM2']
#cell_names = ['HepG2', 'HepG2', 'HepG2', 'HepG2', 'K562', 'K562', 'K562', 'K562']

rbp_name = 'PPIG'
cell_name = 'HepG2'
model_name = 'clip_residualbind'#'clip_conv_net'
ss_type = 'seq'
window = 200

# dataset path
dataset_path = '/media/peter/storage/encode_eclip/eclip_datasets'
dataset_file_path = os.path.join(dataset_path, experiment_name+'_'+str(window)+'.h5')

# set results path
results_path = helper.make_directory('../../results', 'encode_eclip')

# model results path
model_path = helper.make_directory(results_path, model_name)

# path to save model
experiment_name = rbp_name + '_' + cell_name
model_save_path = os.path.join(model_path, experiment_name)

#-----------------------------------------------------------------------------------------
# load dataset
train, valid, test = helper.load_dataset_hdf5(dataset_file_path, ss_type=ss_type)

# get shapes of data
input_shape = list(train['inputs'].shape)
input_shape[0] = None
output_shape = [None, train['targets'].shape[1]]

# build model
genome_model = helper.import_model(model_name)
model_layers, optimization = genome_model(input_shape, output_shape)

# build neural network class
nnmodel = nn.NeuralNet(seed=247)
nnmodel.build_layers(model_layers, optimization)
nnmodel.inspect_layers()

# compile neural trainer
nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=model_save_path)

# initialize session
sess = utils.initialize_session(nnmodel.placeholders)

# train model
data = {'train': train, 'valid': valid}
fit.train_minibatch(sess, nntrainer, data, batch_size=100, num_epochs=200,
                    patience=25, verbose=2, shuffle=True, save_all=False)

sess.close()
