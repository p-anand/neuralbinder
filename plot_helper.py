from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import matplotlib.pyplot as plt

import helper
from deepomics import visualize, saliency

#---------------------------------------------------------------------------------------------------------
# helper functions for visualizations
#---------------------------------------------------------------------------------------------------------



def plot_ensemble_saliency_group(test, predictions, top_indices, models, best_path, plot_path, rbp_name, addon, ss_type='seq', num_plots=5, use_scope=True):
	if len(top_indices) > num_plots:
		plot_indices = top_indices[:num_plots]
	else:
		plot_indices = top_indices

	# sequence to perform saliency analysis
	X = test['inputs'][plot_indices]
	y = test['targets'][plot_indices,0]
	p = predictions[plot_indices]

	input_shape = list(test['inputs'].shape)
	input_shape[0] = None
	output_shape = test['targets'].shape

	mean_saliency, guided_saliency = helper.ensemble_saliency(X, models, best_path, rbp_name, input_shape, output_shape, use_scope)

	for i in range(num_plots):
		if ss_type == 'seq':

			for j in range(len(models)):
				print(plt)
				fig = plt.figure()
				visualize.plot_seq_pos_saliency(np.squeeze(X[i]).T, np.squeeze(guided_saliency[j][i]).T, alphabet='rna')
				output_name = models[j]+'_'+rbp_name+'_'+addon+'_{:.2f}'.format(p[i]) + '_' + '{:.2f}'.format(y[i])
				outfile = os.path.join(plot_path, output_name+'.pdf')
				fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
				plt.close()

			fig = plt.figure()
			visualize.plot_seq_pos_saliency(np.squeeze(X[i]).T, np.squeeze(mean_saliency[i]).T, alphabet='rna')
			output_name = 'ensemble_'+rbp_name+'_'+addon+'_{:.2f}'.format(p[i]) + '_' + '{:.2f}'.format(y[i])
			outfile = os.path.join(plot_path, output_name+'.pdf')
			fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
			plt.close()

		else:

			for j in range(len(models)):
				fig = plt.figure()
				visualize.plot_seq_struct_saliency(np.squeeze(X[i]).T, np.squeeze(guided_saliency[j][i]).T)
				output_name = models[j] +'_'+rbp_name+'_'+addon+'_{:.2f}'.format(p[i]) + '_' + '{:.2f}'.format(y[i])
				outfile = os.path.join(plot_path, output_name+'.pdf')
				fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
				plt.close()

			fig = plt.figure()
			visualize.plot_seq_struct_saliency(np.squeeze(X[i]).T, np.squeeze(mean_saliency[i]).T)
			output_name = 'ensemble_'+rbp_name+'_'+addon+'_{:.2f}'.format(p[i]) + '_' + '{:.2f}'.format(y[i])
			outfile = os.path.join(plot_path, output_name+'.pdf')
			fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
			plt.close()



def plot_saliency_group(test, predictions, top_indices, params, plot_path, name, ss_type='seq', num_plots=5):
	if len(top_indices) > num_plots:
		plot_indices = top_indices[:num_plots]
	else:
		plot_indices = top_indices

	# sequence to perform saliency analysis
	X = test['inputs'][plot_indices]
	y = test['targets'][plot_indices,0]
	p = predictions[plot_indices,0]

	guided_saliency = saliency.guided_backprop(X, layer='output', class_index=0, params=params)
	if guided_saliency.any():
		for i, index in enumerate(plot_indices):
			if ss_type == 'seq':
				fig = plt.figure(figsize=(10,10))
				visualize.plot_seq_pos_saliency(np.squeeze(X[i]).T, np.squeeze(guided_saliency[i]).T, alphabet='rna')
				output_name = name+'_{:.2f}'.format(p[i]) + '_' + '{:.2f}'.format(y[i])
				outfile = os.path.join(plot_path, output_name+'.pdf')
				fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
				plt.close()

			else:
				fig = plt.figure(figsize=(10,10))
				visualize.plot_seq_struct_saliency(np.squeeze(X[i]).T, np.squeeze(guided_saliency[i]).T)
				output_name = name+'_{:.2f}'.format(p[i]) + '_' + '{:.2f}'.format(y[i])
				outfile = os.path.join(plot_path, output_name+'.pdf')
				fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
				plt.close()


def get_percentile_indices(X, Y, lower_percentile, upper_percentile):
	Y_upper_bound = np.percentile(Y, upper_percentile)
	X_upper_bound = np.percentile(X, upper_percentile)
	upper_bound = np.max([Y_upper_bound, X_upper_bound])

	X_lower_bound = np.percentile(X, lower_percentile)
	Y_lower_bound = np.percentile(Y, lower_percentile)
	lower_bound = np.min([X_lower_bound, Y_lower_bound])

	indices = np.where((X >= lower_bound) &
					   (X <= upper_bound) &
					   (Y >= lower_bound) &
					   (Y <= upper_bound))[0]
	return indices[np.argsort((X[indices] - Y[indices])**2)]


def scatter_plot(predictions, experiment, offset=0.3, alpha=0.4, ax=None):

	ax = plt.gca()
	ax.scatter(predictions, experiment, alpha=alpha)
	MIN = np.minimum(np.min(predictions), np.min(experiment))
	MAX = np.maximum(np.max(predictions), np.max(experiment))
	plt.plot([MIN-offset,MAX+offset], [MIN-offset,MAX+offset], '--r')
	plt.ylim([MIN-offset,MAX+offset])
	plt.xlim([MIN-offset,MAX+offset])
	plt.xlabel('Prediction', fontsize=14)
	plt.ylabel('Experiment', fontsize=14)
	ax.tick_params(axis='x', labelsize=14)
	ax.tick_params(axis='y', labelsize=14)



def load_coordinates(coordinate_path, name='test'):
	with open(coordinate_path, 'rb') as f:
		train_coords = cPickle.load(f)
		valid_coords = cPickle.load(f)
		test_coords = cPickle.load(f)

	if name == 'train':
		chrom = train_coords[0]
		start = train_coords[1]
		end = train_coords[2]
		strand_info = train_coords[3]
	elif name == 'valid':
		chrom = valid_coords[0]
		start = valid_coords[1]
		end = valid_coords[2]
		strand_info = valid_coords[3]
	elif name == 'test':
		chrom = test_coords[0]
		start = test_coords[1]
		end = test_coords[2]
		strand_info = test_coords[3]
	return chrom, start, end, strand_info


def plot_clip_ensemble_saliency_group(test, predictions, plot_index, models, best_path, plot_path, experiment_name, ss_type='seq', num_plots=5, use_scope=True):

	# get coordinates of sequences
	chrom, start, end, strand = load_coordinates(coordinate_path, name=name)

	X = test['inputs'][plot_index]

	input_shape = list(test['inputs'].shape)
	input_shape[0] = None
	output_shape = test['targets'].shape

	mean_saliency, guided_saliency = helper.ensemble_saliency(X, models, best_path, experiment_name, input_shape, output_shape, use_scope)

	if guided_saliency.any():
		for i, index in enumerate(plot_index):
			if strand[index] == '+':
				strand_info = 'pos'
			else:
				strand_info = 'neg'
			coordinates = str(chrom[index])+':'+str(start[index])+'-'+str(end[index])+'_'+strand_info

			if ss_type == 'seq':

				for j in range(len(models)):
					print(plt)
					fig = plt.figure()
					visualize.plot_seq_pos_saliency(np.squeeze(X[j]).T, np.squeeze(guided_saliency[j][i]).T, alphabet='rna')
					outfile = os.path.join(save_path, models[j]+'_'+coordinates+'.pdf')
					fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
					plt.close()

				fig = plt.figure()
				visualize.plot_seq_pos_saliency(np.squeeze(X[i]).T, np.squeeze(mean_saliency[i]).T, alphabet='rna')
				outfile = os.path.join(plot_path, 'ensemble_'+coordinates+'.pdf')
				fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
				plt.close()

			else:

				for j in range(len(models)):
					fig = plt.figure()
					visualize.plot_seq_struct_saliency(np.squeeze(X[i]).T, np.squeeze(guided_saliency[j][i]).T)
					outfile = os.path.join(save_path, models[j]+'_'+coordinates+'.pdf')
					fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
					plt.close()

				fig = plt.figure()
				visualize.plot_seq_struct_saliency(np.squeeze(X[i]).T, np.squeeze(mean_saliency[i]).T)
				outfile = os.path.join(plot_path, 'ensemble_'+coordinates+'.pdf')
				fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
				plt.close()
