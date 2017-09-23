from __future__ import print_function 
import os, sys, h5py
import numpy as np
from six.moves import cPickle
from collections import OrderedDict
import matplotlib.pyplot as plt

import tensorflow as tf
import load_data, helper

import tfomics
from tfomics import neuralnetwork as nn
from tfomics import utils, fit, metrics
from tfomics import saliency, visualize
np.random.seed(247)

#------------------------------------------------------------------------------------------

ss_types = ['pu']
models = ['deep_residual_model']
data_path = '/home/peter/invivo_CLIP_200padded_datasets'
results_path = utils.make_directory('../results','CLIP')
save_plot_path = utils.make_directory('../..', 'CLIP_saliency')
# get all clip file names
clip_files = os.listdir(data_path)
clip_names = []
for filename in clip_files:
	clip_names.append(filename[:filename.index('_invivo')])

#------------------------------------------------------------------------------------------

results = []
for i, filename in enumerate(clip_files):

	save_path = utils.make_directory(results_path,clip_names[i])

	for ss_type in ss_types:
		for model in models:
			tf.reset_default_graph() # reset any tensorflow graphs
			np.random.seed(247) # for reproducibility
			tf.set_random_seed(247) # for reproducibility

			# load data
			file_path = os.path.join(data_path, filename)
			train, valid, test = load_data.clip_dataset(file_path, split_ratio=0.9, ss_type=ss_type)

			# get shapes
			input_shape = list(train['inputs'].shape)
			input_shape[0] = None
			output_shape = train['targets'].shape

			# set output file paths
			output_name = model + '_' + ss_type
			file_path = os.path.join(save_path, output_name)

			# load model
			if model == 'conv_net':
				from models import conv_net_clip as genome_model
			elif model == 'deep_residual_model':
				from models import deep_residual_model_clip as genome_model
			
			model_layers, optimization = genome_model.model(input_shape, output_shape)

			# build neural network class
			nnmodel = nn.NeuralNet(seed=247)
			nnmodel.build_layers(model_layers, optimization)
			#nnmodel.inspect_layers()

			# compile neural trainer
			nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

			# initialize session
			sess = utils.initialize_session(nnmodel.placeholders)

			# load best model
			nntrainer.set_best_parameters(sess)

			# test model
			loss, mean_vals, std_vals = nntrainer.test_model(sess, test, batch_size=128, name='test', verbose=1)

			predictions = nntrainer.get_activations(sess, test, 'output')

			scores, curves = metrics.roc(test['targets'], predictions)
			results.append((clip_names[i], model, ss_type, mean_vals, curves))

			# get predictions, sort them, and find the indices of true class
			predictions = nntrainer.get_activations(sess, data=test, layer='output',)
			true_index = np.where(test['targets'] == 1)[0]
			max_index = true_index[np.argsort(predictions[true_index,0])[::-1]]

			# plot and save saliency of top predicted sequences
			plot_path = utils.make_directory(save_plot_path, output_name)
			
			num_plots = 20
			for j, index in enumerate(max_index[:num_plots]):
				# sequence to perform saliency analysis
				
				X = np.expand_dims(test['inputs'][index], axis=0)

				# parameters for saliency analysis
				params = {'genome_model': genome_model.model, 
						  'input_shape': input_shape, 
						  'output_shape': output_shape, 
						  'model_path': file_path+'_best.ckpt',
						  'optimization': optimization
						 }

				plot_range = range(75,125)
				# guided backprop saliency
				guided_saliency = saliency.guided_backprop(X, layer='output', class_index=0, params=params)

				fig = plt.figure(figsize=(10,10))
				plt = visualize.plot_seq_struct_saliency(np.squeeze(X[0]).T[:,plot_range], np.squeeze(guided_saliency).T[:,plot_range])
				outfile = os.path.join(plot_path, clip_names[i]+'_' + str(index) + '.pdf')
				fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight') 
				plt.close()
				