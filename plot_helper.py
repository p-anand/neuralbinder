from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from deepomics import visualize, saliency

#---------------------------------------------------------------------------------------------------------
# helper functions for visualizations
#---------------------------------------------------------------------------------------------------------

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
				output_name = name+'_seq_guided_{:.2f}'.format(p[i]) + '_' + '{:.2f}'.format(y[i])
				outfile = os.path.join(plot_path, output_name+'.pdf')
				fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight') 
				plt.close()

			else:
				fig = plt.figure(figsize=(10,10))
				visualize.plot_seq_struct_saliency(np.squeeze(X[i]).T, np.squeeze(guided_saliency[i]).T)
				output_name = name+'_'+ss_type+'_guided_{:.2f}'.format(p[i]) + '_' + '{:.2f}'.format(y[i])
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

	
def plot_clip_saliency(sess, test, plot_index, params, coordinate_path, save_path, name='test'):

	# get coordinates of sequences
	chrom, start, end, strand = helper.load_coordinates(coordinate_path, name=name)

	X = test['inputs'][plot_index]

	# guided backprop saliency
	guided_saliency = saliency.guided_backprop(X, layer='output', class_index=0, params=params)

	if guided_saliency.any():
		# plot saliency
		for i, index in enumerate(plot_index):
			
			fig = plt.figure(figsize=(100,10))
			if X.shape[3] == 4:
				visualize.plot_seq_pos_saliency(np.squeeze(X[i]).T, np.squeeze(guided_saliency[i]).T, alphabet='rna')
			elif X.shape[4] == 6:
				visualize.plot_seq_struct_saliency(np.squeeze(X[i]).T, np.squeeze(guided_saliency[i]).T)

			if strand[index] == '+':
				strand_info = 'pos'
			else:
				strand_info = 'neg'

			coordinates = str(chrom[index])+':'+str(start[index])+'-'+str(end[index])+'_'+strand_info
			outfile = os.path.join(save_path, coordinates+'.pdf')
			fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')

			plt.close()



