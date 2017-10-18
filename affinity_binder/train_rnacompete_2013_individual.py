#---------------------------------------------------------------------------------------
"""
Summary: Train deep learning models on a single RNAcompete_2013 dataset.
"""
#---------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
import tensorflow as tf

sys.path.append('..')
import helper
from deepomics import neuralnetwork as nn
from deepomics import utils, fit

#---------------------------------------------------------------------------------------

#0 conv_net seq
#13 conv_net seq
#96 conv_net pu

rbp_index = 219
num_epochs = 100
batch_size = 100

# different deep learning models to try out
model = 'affinity_conv_net'
normalize_method = 'log_norm'   # 'clip_norm'
ss_type = 'seq'


data_path = '../../data/RNAcompete_2013/rnacompete2013.h5'
results_path = helper.make_directory('../../results', 'test')

#---------------------------------------------------------------------------------------

# get list of rnacompete experiments
experiments = helper.get_experiments_hdf5(data_path)
experiment = experiments[rbp_index]
print('Analyzing: '+ experiment)

# load rbp dataset
train, valid, test = helper.load_dataset_hdf5(data_path, ss_type=ss_type, rbp_index=rbp_index)

# process rbp dataset
train, valid, test = helper.process_data(train, valid, test, method=normalize_method)

# get shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None
output_shape = train['targets'].shape

tf.reset_default_graph() # reset any tensorflow graphs
#np.random.seed(247) # for reproducibilitjy
#tf.set_random_seed(247) # for reproducibility

# load model
genome_model = helper.import_model(model)
model_layers, optimization = genome_model.model(input_shape, output_shape)

# build neural network class
nnmodel = nn.NeuralNet()
nnmodel.build_layers(model_layers, optimization, use_scope=False)

# loop over different secondary structure contexts
sstype_path = helper.make_directory(results_path, normalize_method+'_'+ss_type)
model_path = helper.make_directory(sstype_path, model)
file_path = os.path.join(model_path, experiment)
nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

# initialize session
sess = utils.initialize_session(nnmodel.placeholders)

# fit model
data = {'train': train, 'valid': valid}
fit.train_minibatch(sess, nntrainer, data, batch_size=batch_size, num_epochs=num_epochs,
	  patience=20, verbose=2, shuffle=True, save_all=False)

sess.close()
