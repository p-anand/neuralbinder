#---------------------------------------------------------------------------------------
"""
Summary: Train a simple model on the saliency maps of individual models for
top-scoring sequences to identify learned motifs in the RNAcompete 2013 dataset.
"""
#---------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import neuralbinder.neuralbindhelpers.helper as helper
from neuralbinder.deepomics import neuralnetwork as nn
from neuralbinder.deepomics import utils, fit, metrics, saliency, visualize, init

#---------------------------------------------------------------------------------------

num_saliencies = [200]
num_filters = 5

# different deep learning models to try out
models = ['affinity_conv_net', 'affinity_residualbind', 'affinity_all_conv_net']
normalize_method = 'log_norm'   # 'clip_norm'
ss_types = ['seq', 'pu']

data_path = neuralbinder.get_datasets('rnacompete2013.h5')
trained_path = '../../results/RNAcompete_2013'
results_path = utils.make_directory(trained_path, 'motifs')


#---------------------------------------------------------------------------------------
# classifier model

def classifier_model(input_shape, output_shape, num_filters):

	# create model
	layer1 = {'layer': 'input', #41
			'input_shape': input_shape
			}
	layer2 = {'layer': 'conv1d',
			'num_filters': num_filters,
			'filter_size': 11,
			'activation': 'sigmoid',
			'W': init.HeUniform(),
			'padding': 'SAME',
			'global_pool': 'max',
			}
	layer3 = {'layer': 'reshape',
			  'reshape': [-1, num_filters],
			}
	layer4 = {'layer': 'mean_pool',
			}
	layer5 = {'layer': 'reshape',
			  'reshape': [-1, 1],
			}
	#from tfomics import build_network
	model_layers = [layer1, layer2, layer3, layer4, layer5]

	# optimization parameters
	optimization = {"objective": "binary",
				  "optimizer": "adam",
				  "learning_rate": 0.001,
				  #"momentum": 0.9,
				  #"l2": 1e-6,
				  }
	return model_layers, optimization


#---------------------------------------------------------------------------------------

# get list of rnacompete experiments
experiments = helper.get_experiments_hdf5(data_path)

# loop over different secondary structure contexts
for ss_type in ss_types:
	print('input data: ' + ss_type)
	sstype_path = os.path.join(trained_path, normalize_method+'_'+ss_type)

	# path to save parameters of saliency classifier
	classifier_sstype_path = helper.make_directory(results_path, normalize_method+'_'+ss_type)

	# loop over different models
	for model in models:
		print('model: ' + model)
		model_path = os.path.join(sstype_path, model)

		classifier_model_path = helper.make_directory(classifier_sstype_path, model)

		# loop over different RNA binding proteins
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
			nnmodel.build_layers(model_layers, optimization)


			# compile neural trainer
			file_path = os.path.join(model_path, experiment)
			nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

			# initialize session
			sess = utils.initialize_session(nnmodel.placeholders)

			# load best parameters
			nntrainer.set_best_parameters(sess)

			# get predictions
			predictions = nntrainer.get_activations(sess, train, layer='output')
			max_index = np.argsort(predictions[:,0])[::-1]

			# close tensorflow session
			sess.close()

			# directories to store classifier model and motifs
			motif_path = helper.make_directory(classifier_model_path, experiment)
			params_path = helper.make_directory(motif_path, 'parameters')
			for num_saliency in num_saliencies:

				# get saliency for highest predictions
				X = train['inputs'][max_index[:num_saliency]]

				# parameters for saliency analysis
				params = {'genome_model': genome_model,
						  'input_shape': input_shape,
						  'output_shape': output_shape,
						  'model_path': file_path+'_best.ckpt',
						  'optimization': optimization
						 }

				# guided backprop saliency
				guided_saliency = saliency.guided_backprop(X, layer='output', class_index=0, params=params)

				# shuffle saliency for background data
				background = []
				for i in range(num_saliency):
					shuffle = np.random.permutation(41)
					background.append([guided_saliency[i,shuffle,:,:]])
				background = np.vstack(background)

				# merge datasets
				inputs = np.vstack([guided_saliency, background])
				targets = np.vstack([np.ones((num_saliency,1)), np.zeros((num_saliency,1))])
				shuffle = np.random.permutation(inputs.shape[0])
				new_train = {'inputs': inputs[shuffle], 'targets': targets[shuffle]}

				# load classifier model
				model_layers, optimization = classifier_model(input_shape, output_shape, num_filters)

				# build neural network class
				nnmodel = nn.NeuralNet(seed=247)
				nnmodel.build_layers(model_layers, optimization)

				# compile neural trainer
				classifier_path = os.path.join(params_path, 'saliency_classifier_'+str(num_saliency))
				nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=classifier_path)

				# initialize session
				sess = utils.initialize_session(nnmodel.placeholders)

				# fit data
				data = {'train': new_train}
				fit.train_minibatch(sess, nntrainer, data, batch_size=32, num_epochs=500,
									  patience=20, verbose=2, shuffle=True, save_all=False)


				# plot mean activations for each filter
				data={'inputs': guided_saliency}
				fmaps = nntrainer.get_activations(sess, data, layer='conv1d_0_active')
				mean_fmap = np.squeeze(np.mean(fmaps, axis=0))
				fig = plt.figure()
				plt.plot(range(1,42), mean_fmap)
				plt.xticks()
				labels = range(1, num_filters+2)
				plt.legend(labels, fontsize=16, frameon=False, bbox_to_anchor=(1, 1))
				ax = plt.gca();
				plt.setp(ax.get_xticklabels(),fontsize=16);
				plt.setp(ax.get_yticklabels(),fontsize=16);
				plt.xlabel('Position (nt)', fontsize=16)
				plt.ylabel('Activation frequency ', fontsize=16)
				outfile = os.path.join(motif_path, str(num_saliency)+'_activations.pdf')
				fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
				plt.close()

				# extract pwm from activation alignments and plot pwms
				pwm = visualize.generate_pwm(sess, nntrainer, np.squeeze(X), guided_saliency, window=16, layer='conv1d_0_active')
				for i, W in enumerate(pwm):
					if not np.isnan(W).any():
						logo = visualize.seq_logo(W.T[:4,:], height=300, nt_width=100, alphabet='rna')

						if W.shape[1] > 4:
							struct_logo = visualize.seq_logo(W.T[4:,:], height=600, nt_width=100, alphabet='pu')
							logo = np.vstack([logo, struct_logo])

						fig = plt.figure()
						plt.imshow(logo)
						plt.axis('off')
						outfile = os.path.join(motif_path, str(num_saliency)+'_filter_'+ str(i+1)+'_logo.pdf')
						fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
						plt.close()

						# clip sequence logos
						start, end = helper.get_clip_indices(W.T, threshold=0.2, window=2)
						logo = visualize.seq_logo(W.T[:4,start:end], height=300, nt_width=100, alphabet='rna')
						if W.shape[1] > 4:
							struct_logo = visualize.seq_logo(W.T[4:,start:end], height=600, nt_width=100, alphabet='pu')
							logo = np.vstack([logo, struct_logo])
						fig = plt.figure()
						plt.imshow(logo)
						plt.axis('off')
						outfile = os.path.join(motif_path, str(num_saliency)+'_clip_filter_'+ str(i+1)+'_logo.pdf')
						fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
						plt.close()


						fig = plt.figure()
						plt = visualize.filter_heatmap(W.T, norm=True, cmap='hot_r')
						outfile = os.path.join(motif_path, str(num_saliency)+'_filter_'+ str(i+1)+'_heatmap.pdf')
						fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
						plt.close()


				sess.close()

				"""
				# plot filters directly
				W = nnmodel.get_parameters(sess)[0]
				num_filters = W.shape[-1]
				for i in range(num_filters):
					plt.figure()
					logo = visualize.seq_logo(utils.normalize_pwm(np.squeeze(W[:,:,:,i].T)[:4,:],factor=3), height=300, nt_width=100, alphabet='rna')
					if W.shape[-2] > 4:
						struct_logo = visualize.seq_logo(utils.normalize_pwm(np.squeeze(W[:,:,:,i].T)[4:,:],factor=3), height=600, nt_width=100, alphabet='pu')
						logo = np.vstack([logo, struct_logo])

					plt.figure()
					plt.imshow(logo)
					plt.axis('off')

					plt.figure()
					plt = visualize.filter_heatmap(np.squeeze(W[:,:,:,i].T), norm=None, cmap='seismic', cbar_norm=False)
				"""
